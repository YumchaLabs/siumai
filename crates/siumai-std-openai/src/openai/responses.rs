//! OpenAI Responses API standard (request/response/streaming skeleton).
//!
//! This module provides a minimal, provider-agnostic standard for
//! shaping OpenAI Responses API requests and responses on top of the
//! core `siumai-core` `ResponsesInput` / `ResponsesResult` types.
//!
//! The initial version focused on request shaping and a very small
//! response mapping. Streaming is gradually being migrated from the
//! aggregator into this standard via `ChatStreamEventCore`.

use serde::{Deserialize, Serialize};
use siumai_core::error::LlmError;
use siumai_core::execution::responses::{
    ResponsesInput, ResponsesRequestTransformer, ResponsesResponseTransformer, ResponsesResult,
};
use siumai_core::execution::streaming::{ChatStreamEventConverterCore, ChatStreamEventCore};
use siumai_core::types::FinishReasonCore;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// OpenAI Responses API standard entry point.
///
/// This is intentionally minimal: it delegates most of the message →
/// JSON shaping logic to helpers that mirror the existing aggregator
/// behaviour, but operates purely on core types.
#[derive(Clone, Default)]
pub struct OpenAiResponsesStandard;

impl OpenAiResponsesStandard {
    /// Create a new Responses standard instance.
    pub fn new() -> Self {
        Self
    }

    /// Create a request transformer for Responses API.
    pub fn create_request_transformer(
        &self,
        provider_id: &str,
    ) -> Arc<dyn ResponsesRequestTransformer> {
        Arc::new(OpenAiResponsesRequestTx {
            provider_id: provider_id.to_string(),
        })
    }

    /// Create a response transformer for Responses API.
    ///
    /// Initial implementation is intentionally thin and only maps the
    /// top-level `output` and `usage` fields; provider crates can
    /// extend this as needed.
    pub fn create_response_transformer(
        &self,
        provider_id: &str,
    ) -> Arc<dyn ResponsesResponseTransformer> {
        Arc::new(OpenAiResponsesResponseTx {
            provider_id: provider_id.to_string(),
        })
    }

    /// Create a streaming converter for the Responses API.
    ///
    /// This maps OpenAI Responses SSE events into the core streaming
    /// model (`ChatStreamEventCore`), emitting:
    /// - `StreamStart` once per stream
    /// - `ContentDelta` for text deltas
    /// - `ToolCallDelta` for tool call deltas
    /// - `UsageUpdate` for usage events
    /// - `Error` for error events
    /// - `StreamEnd` when a `response.completed` event is observed.
    pub fn create_stream_converter(
        &self,
        provider_id: &str,
    ) -> Arc<dyn ChatStreamEventConverterCore> {
        Arc::new(OpenAiResponsesStreamConverter::new(provider_id.to_string()))
    }
}

/// Normalized representation of a single tool call in the Responses output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiResponsesToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Normalized representation of Responses output (text + tool calls).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpenAiResponsesOutput {
    pub text: String,
    pub tool_calls: Vec<OpenAiResponsesToolCall>,
}

/// Parse an OpenAI Responses output object into a normalized structure.
///
/// This helper mirrors the aggregator-level Responses transformer behavior
/// but lives in the std layer so that other languages/bindings can reuse the
/// same semantics.
pub fn parse_responses_output(root: &serde_json::Value) -> OpenAiResponsesOutput {
    let mut text_content = String::new();
    let mut tool_calls = Vec::new();

    if let Some(output) = root.get("output").and_then(|v| v.as_array()) {
        for item in output {
            // Collect text content
            if let Some(parts) = item.get("content").and_then(|c| c.as_array()) {
                for p in parts {
                    if let Some(t) = p.get("text").and_then(|v| v.as_str()) {
                        if !text_content.is_empty() {
                            text_content.push('\n');
                        }
                        text_content.push_str(t);
                    }
                }
            }

            // Collect tool calls (nested function object or flattened)
            if let Some(calls) = item.get("tool_calls").and_then(|tc| tc.as_array()) {
                for call in calls {
                    let id = call
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let (name, arguments) = if let Some(f) = call.get("function") {
                        (
                            f.get("name")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string(),
                            f.get("arguments")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string(),
                        )
                    } else {
                        (
                            call.get("name")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string(),
                            call.get("arguments")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string(),
                        )
                    };
                    if !name.is_empty() {
                        let args_value = serde_json::from_str(&arguments)
                            .unwrap_or(serde_json::Value::String(arguments));
                        tool_calls.push(OpenAiResponsesToolCall {
                            id,
                            name,
                            arguments: args_value,
                        });
                    }
                }
            }
        }
    }

    OpenAiResponsesOutput {
        text: text_content,
        tool_calls,
    }
}

#[derive(Clone)]
struct OpenAiResponsesRequestTx {
    provider_id: String,
}

impl ResponsesRequestTransformer for OpenAiResponsesRequestTx {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn transform_responses(&self, req: &ResponsesInput) -> Result<serde_json::Value, LlmError> {
        // Base shape: model + input[]
        let mut body = serde_json::json!({
            "model": req.model,
            "input": req.input,
        });

        // Flatten extra config into the top-level body (mirrors current
        // behaviour in OpenAiSpec::chat_before_send for Responses API).
        if let Some(obj) = body.as_object_mut() {
            for (k, v) in &req.extra {
                obj.insert(k.clone(), v.clone());
            }
        }

        Ok(body)
    }
}

#[derive(Clone)]
struct OpenAiResponsesResponseTx {
    provider_id: String,
}

impl ResponsesResponseTransformer for OpenAiResponsesResponseTx {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn transform_responses_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<ResponsesResult, LlmError> {
        // Minimal mapping: capture entire raw JSON as `output`, and
        // attempt to extract usage stats if present. This mirrors the
        // aggregator-level OpenAI Responses transformer logic so that
        // usage semantics are consistent across core/std/provider
        // layers.
        //
        // The `raw` payload may be either the full top-level object or
        // the nested `response` field; we normalize `output` to the
        // nested `response` object so that callers can rely on a
        // consistent shape.
        let root = raw.get("response").unwrap_or(raw);
        let mut result = ResponsesResult {
            output: root.clone(),
            usage: None,
            finish_reason: None,
            metadata: Default::default(),
        };

        if let Some(usage) = root.get("usage") {
            // The Responses API can expose usage under several field
            // names (snake_case and camelCase). We follow the same
            // precedence rules as the aggregator:
            //
            // - prompt/input tokens:
            //   - input_tokens | prompt_tokens | inputTokens
            // - completion/output tokens:
            //   - output_tokens | completion_tokens | outputTokens
            // - total tokens:
            //   - total_tokens | totalTokens
            let prompt_tokens = usage
                .get("input_tokens")
                .or_else(|| usage.get("prompt_tokens"))
                .or_else(|| usage.get("inputTokens"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;

            let completion_tokens = usage
                .get("output_tokens")
                .or_else(|| usage.get("completion_tokens"))
                .or_else(|| usage.get("outputTokens"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;

            let total_tokens = usage
                .get("total_tokens")
                .or_else(|| usage.get("totalTokens"))
                .and_then(|v| v.as_u64())
                .unwrap_or((prompt_tokens + completion_tokens) as u64)
                as u32;

            result.usage = Some(siumai_core::execution::responses::ResponsesUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens,
            });
        }

        // Normalize finish reason from stop_reason / finish_reason into
        // a core enum, applying the same canonicalization rules used in
        // the aggregator (max_tokens → length, tool_use → tool_calls,
        // safety → content_filter).
        let raw_reason = root
            .get("stop_reason")
            .or_else(|| root.get("finish_reason"))
            .and_then(|v| v.as_str());

        let canon_reason = match raw_reason {
            Some("max_tokens") => Some("length"),
            Some("tool_use") | Some("function_call") => Some("tool_calls"),
            Some("safety") => Some("content_filter"),
            Some(other) => Some(other),
            None => None,
        };

        result.finish_reason = FinishReasonCore::from_str(canon_reason);

        Ok(result)
    }
}

#[derive(Clone)]
struct OpenAiResponsesStreamConverter {
    provider_id: String,
    started: Arc<AtomicBool>,
    ended: Arc<AtomicBool>,
}

impl OpenAiResponsesStreamConverter {
    fn new(provider_id: String) -> Self {
        Self {
            provider_id,
            started: Arc::new(AtomicBool::new(false)),
            ended: Arc::new(AtomicBool::new(false)),
        }
    }

    fn convert_delta_event(&self, json: &serde_json::Value) -> Option<ChatStreamEventCore> {
        // Handle delta as plain text or delta.content
        if let Some(delta) = json.get("delta") {
            // Case 1: delta is a plain string (response.output_text.delta)
            if let Some(s) = delta.as_str() {
                if !s.is_empty() {
                    return Some(ChatStreamEventCore::ContentDelta {
                        delta: s.to_string(),
                        index: None,
                    });
                }
            }
            // Case 2: delta.content is a string (message.delta simplified)
            if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                return Some(ChatStreamEventCore::ContentDelta {
                    delta: content.to_string(),
                    index: None,
                });
            }

            // Handle tool_calls delta (first item only; downstream can coalesce)
            if let Some(tool_calls) = delta.get("tool_calls").and_then(|tc| tc.as_array()) {
                if let Some((index, tool_call)) = tool_calls.iter().enumerate().next() {
                    let id = tool_call
                        .get("id")
                        .and_then(|id| id.as_str())
                        .map(|s| s.to_string());

                    let function_name = tool_call
                        .get("function")
                        .and_then(|func| func.get("name"))
                        .and_then(|n| n.as_str())
                        .map(|s| s.to_string());

                    let arguments_delta = tool_call
                        .get("function")
                        .and_then(|func| func.get("arguments"))
                        .and_then(|a| a.as_str())
                        .map(|s| s.to_string());

                    return Some(ChatStreamEventCore::ToolCallDelta {
                        id,
                        function_name,
                        arguments_delta,
                        index: Some(index),
                    });
                }
            }
        }

        // Handle usage updates with both snake_case and camelCase fields
        if let Some(usage) = json
            .get("usage")
            .or_else(|| json.get("response").and_then(|r| r.get("usage")))
        {
            let prompt_tokens = usage
                .get("prompt_tokens")
                .or_else(|| usage.get("input_tokens"))
                .or_else(|| usage.get("inputTokens"))
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as u32)
                .unwrap_or(0);
            let completion_tokens = usage
                .get("completion_tokens")
                .or_else(|| usage.get("output_tokens"))
                .or_else(|| usage.get("outputTokens"))
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as u32)
                .unwrap_or(0);
            let total_tokens = usage
                .get("total_tokens")
                .or_else(|| usage.get("totalTokens"))
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as u32)
                .unwrap_or(prompt_tokens.saturating_add(completion_tokens));

            return Some(ChatStreamEventCore::UsageUpdate {
                prompt_tokens,
                completion_tokens,
                total_tokens,
            });
        }

        None
    }

    fn convert_function_call_arguments_delta(
        &self,
        json: &serde_json::Value,
    ) -> Option<ChatStreamEventCore> {
        // Handle response.function_call_arguments.delta events
        let delta = json.get("delta").and_then(|d| d.as_str())?;
        let item_id = json.get("item_id").and_then(|id| id.as_str()).unwrap_or("");
        let output_index = json
            .get("output_index")
            .and_then(|idx| idx.as_u64())
            .unwrap_or(0) as usize;

        Some(ChatStreamEventCore::ToolCallDelta {
            id: Some(item_id.to_string()),
            function_name: None, // Function name is set in the initial item.added event
            arguments_delta: Some(delta.to_string()),
            index: Some(output_index),
        })
    }

    fn convert_output_item_added(&self, json: &serde_json::Value) -> Option<ChatStreamEventCore> {
        // Handle response.output_item.added events for function calls
        let item = json.get("item")?;
        if item.get("type").and_then(|t| t.as_str()) != Some("function_call") {
            return None;
        }

        let id = item.get("call_id").and_then(|id| id.as_str()).unwrap_or("");
        let function_name = item.get("name").and_then(|name| name.as_str());
        let output_index = json
            .get("output_index")
            .and_then(|idx| idx.as_u64())
            .unwrap_or(0) as usize;

        Some(ChatStreamEventCore::ToolCallDelta {
            id: Some(id.to_string()),
            function_name: function_name.map(|s| s.to_string()),
            arguments_delta: None, // Arguments will come in subsequent delta events
            index: Some(output_index),
        })
    }
}

impl ChatStreamEventConverterCore for OpenAiResponsesStreamConverter {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn convert_event(
        &self,
        event: eventsource_stream::Event,
    ) -> Vec<Result<ChatStreamEventCore, LlmError>> {
        let mut out = Vec::new();
        let data_raw = event.data.trim();
        if data_raw.is_empty() {
            return out;
        }

        // Do not treat [DONE] specially here; Responses API uses
        // `response.completed` events to signal logical completion.
        if data_raw == "[DONE]" {
            return out;
        }

        // Emit a StreamStart event once per stream, on the first non-empty,
        // non-[DONE] chunk.
        if self
            .started
            .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        {
            out.push(Ok(ChatStreamEventCore::StreamStart {}));
        }

        let event_name = event.event.as_str();

        // Handle response.completed explicitly: this is the primary logical
        // end-of-stream signal for the Responses API.
        if event_name == "response.completed" {
            let json: serde_json::Value = match serde_json::from_str(data_raw) {
                Ok(v) => v,
                Err(e) => {
                    return vec![Err(LlmError::ParseError(format!(
                        "Failed to parse completed event JSON: {e}"
                    )))];
                }
            };

            // Attempt to derive a finish_reason from the completed payload
            let reason = json.get("response").and_then(|r| {
                r.get("stop_reason")
                    .or_else(|| r.get("finish_reason"))
                    .and_then(|v| v.as_str())
            });

            let canon_reason = match reason {
                Some("max_tokens") => Some("length"),
                Some("tool_use") | Some("function_call") => Some("tool_calls"),
                Some("safety") => Some("content_filter"),
                Some(other) => Some(other),
                None => None,
            };

            let finish_reason = FinishReasonCore::from_str(canon_reason);

            // Mark that we have emitted a StreamEnd and avoid duplicates.
            if self
                .ended
                .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                out.push(Ok(ChatStreamEventCore::StreamEnd { finish_reason }));
            }
            return out;
        }

        // Parse JSON for non-completed events
        let json: serde_json::Value = match serde_json::from_str(data_raw) {
            Ok(v) => v,
            Err(e) => {
                return vec![Err(LlmError::ParseError(format!(
                    "Failed to parse SSE JSON: {e}"
                )))];
            }
        };

        match event_name {
            // Text/tool-call delta & usage events
            "response.output_text.delta"
            | "response.tool_call.delta"
            | "response.function_call.delta"
            | "response.usage" => {
                if let Some(evt) = self.convert_delta_event(&json) {
                    out.push(Ok(evt));
                }
            }
            // Error events
            "response.error" => {
                let msg = json
                    .get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(|m| m.as_str())
                    .unwrap_or("Unknown error")
                    .to_string();
                out.push(Ok(ChatStreamEventCore::Error { error: msg }));
            }
            // Function call arguments delta
            "response.function_call_arguments.delta" => {
                if let Some(evt) = self.convert_function_call_arguments_delta(&json) {
                    out.push(Ok(evt));
                }
            }
            // Output item added (initial function call)
            "response.output_item.added" => {
                if let Some(evt) = self.convert_output_item_added(&json) {
                    out.push(Ok(evt));
                }
            }
            // Fallback: attempt generic delta/usage mapping regardless of event name
            _ => {
                if let Some(evt) = self.convert_delta_event(&json) {
                    out.push(Ok(evt));
                }
            }
        }

        out
    }

    fn handle_stream_end(&self) -> Option<Result<ChatStreamEventCore, LlmError>> {
        // For Responses API we rely on `response.completed` as the primary
        // logical completion signal. To avoid emitting duplicate or synthetic
        // StreamEnd events on [DONE], this returns None. This matches the
        // behaviour of the aggregator-level Responses converter.
        None
    }
}
