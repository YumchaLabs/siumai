//! Axum-specific server adapters
//!
//! This module provides utilities to convert `ChatStream` into Axum-compatible responses.
//!
//! ## Features
//!
//! - **SSE Response**: `to_sse_response()` converts `ChatStream` to `Sse<impl Stream>`
//! - **Text Response**: `to_text_stream()` converts `ChatStream` to plain text stream
//! - **Error Handling**: Automatic error masking for production environments
//! - **Type Safety**: Strongly typed with Axum's SSE types
//!
//! ## Example
//!
//! ```rust,ignore
//! use axum::{Router, routing::get, response::sse::Sse};
//! use siumai_extras::server::axum::{to_sse_response, SseOptions};
//! use siumai::prelude::unified::ChatStream;
//!
//! async fn chat_handler(stream: ChatStream) -> Sse<impl futures::Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>>> {
//!     to_sse_response(stream, SseOptions::production())
//! }
//!
//! #[tokio::main]
//! async fn main() {
//!     let app = Router::new().route("/chat", get(chat_handler));
//!     // ... serve the app
//! }
//! ```

use std::convert::Infallible;
use std::pin::Pin;

use axum::response::sse::{Event, Sse};
use axum::{body::Body, http::header, response::Response};
use futures::{Stream, StreamExt};

use siumai::prelude::unified::{
    ChatResponse, ChatStream, ChatStreamEvent, ContentPart, FinishReason, LlmError,
};

/// Target SSE wire format for stream transcoding helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetSseFormat {
    /// OpenAI Responses API SSE (`response.*` events).
    OpenAiResponses,
    /// OpenAI Chat Completions SSE (`chat.completion.chunk` + `[DONE]`).
    OpenAiChatCompletions,
    /// Anthropic Messages API SSE (`message_*` events).
    AnthropicMessages,
    /// Gemini/Vertex GenerateContent SSE (`data: { candidates: ... }` frames).
    GeminiGenerateContent,
}

/// Target JSON wire format for non-streaming response transcoding helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetJsonFormat {
    /// OpenAI Responses API JSON response.
    OpenAiResponses,
    /// OpenAI Chat Completions JSON response.
    OpenAiChatCompletions,
    /// Anthropic Messages API JSON response.
    AnthropicMessages,
    /// Gemini/Vertex GenerateContent JSON response.
    GeminiGenerateContent,
}

/// Options for transcoding a `ChatResponse` into a provider JSON response body.
#[derive(Debug, Clone)]
pub struct TranscodeJsonOptions {
    /// Whether to pretty-print the JSON output.
    pub pretty: bool,
}

impl Default for TranscodeJsonOptions {
    fn default() -> Self {
        Self { pretty: false }
    }
}

/// Options for transcoding a `ChatStream` into a provider SSE wire format.
#[derive(Debug, Clone)]
pub struct TranscodeSseOptions {
    /// Controls lossy fallback for v3 parts that do not have a native representation
    /// in the target protocol stream.
    pub v3_unsupported_part_behavior: siumai::experimental::streaming::V3UnsupportedPartBehavior,
    /// Whether to bridge multiplexed Vercel-aligned tool parts into richer OpenAI Responses
    /// output_item frames (adds rawItem scaffolding when possible).
    pub bridge_openai_responses_stream_parts: bool,
    /// Whether to serialize v3 tool results as Gemini `functionResponse` frames (gateway-only).
    pub gemini_emit_function_response_tool_results: bool,
}

impl Default for TranscodeSseOptions {
    fn default() -> Self {
        Self {
            v3_unsupported_part_behavior:
                siumai::experimental::streaming::V3UnsupportedPartBehavior::Drop,
            bridge_openai_responses_stream_parts: true,
            gemini_emit_function_response_tool_results: false,
        }
    }
}

impl TranscodeSseOptions {
    /// Strict/default transcoding options (drops unsupported v3 parts).
    pub fn strict() -> Self {
        Self::default()
    }

    /// Lossy transcoding options (downgrades unsupported v3 parts to text deltas when possible).
    pub fn lossy_text() -> Self {
        Self {
            v3_unsupported_part_behavior:
                siumai::experimental::streaming::V3UnsupportedPartBehavior::AsText,
            ..Self::default()
        }
    }
}

fn openai_finish_reason(reason: Option<&FinishReason>) -> Option<&'static str> {
    match reason? {
        FinishReason::Stop | FinishReason::StopSequence => Some("stop"),
        FinishReason::Length => Some("length"),
        FinishReason::ContentFilter => Some("content_filter"),
        FinishReason::ToolCalls => Some("tool_calls"),
        FinishReason::Error => Some("error"),
        FinishReason::Unknown => None,
        FinishReason::Other(_) => None,
    }
}

fn anthropic_stop_reason(reason: Option<&FinishReason>) -> Option<&'static str> {
    match reason? {
        FinishReason::Stop | FinishReason::StopSequence => Some("end_turn"),
        FinishReason::Length => Some("max_tokens"),
        FinishReason::ContentFilter => Some("stop_sequence"),
        FinishReason::ToolCalls => Some("tool_use"),
        FinishReason::Error => Some("end_turn"),
        FinishReason::Unknown => None,
        FinishReason::Other(_) => None,
    }
}

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

/// Convert a unified `ChatResponse` into a provider-native JSON response body (best-effort).
///
/// Notes:
/// - This is primarily intended for gateways/proxies that expose multiple protocol surfaces.
/// - Some provider response shapes require fields that are not available in `ChatResponse`
///   (e.g. detailed output item IDs); those fields are filled with reasonable defaults.
pub fn transcode_chat_response_to_json(
    response: &ChatResponse,
    target: TargetJsonFormat,
) -> serde_json::Value {
    fn usage_json(u: &siumai::prelude::unified::Usage) -> serde_json::Value {
        let mut out = serde_json::Map::new();
        out.insert(
            "prompt_tokens".to_string(),
            serde_json::Value::Number(serde_json::Number::from(u.prompt_tokens)),
        );
        out.insert(
            "completion_tokens".to_string(),
            serde_json::Value::Number(serde_json::Number::from(u.completion_tokens)),
        );
        out.insert(
            "total_tokens".to_string(),
            serde_json::Value::Number(serde_json::Number::from(u.total_tokens)),
        );
        if let Some(details) = u.prompt_tokens_details.as_ref() {
            out.insert(
                "prompt_tokens_details".to_string(),
                serde_json::to_value(details).unwrap_or(serde_json::Value::Null),
            );
        }
        if let Some(details) = u.completion_tokens_details.as_ref() {
            out.insert(
                "completion_tokens_details".to_string(),
                serde_json::to_value(details).unwrap_or(serde_json::Value::Null),
            );
        }
        serde_json::Value::Object(out)
    }

    fn tool_calls_json(parts: &[&ContentPart]) -> serde_json::Value {
        let calls: Vec<serde_json::Value> = parts
            .iter()
            .filter_map(|p| match p {
                ContentPart::ToolCall {
                    tool_call_id,
                    tool_name,
                    arguments,
                    ..
                } => {
                    let args_str =
                        serde_json::to_string(arguments).unwrap_or_else(|_| arguments.to_string());
                    Some(serde_json::json!({
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": args_str,
                        }
                    }))
                }
                _ => None,
            })
            .collect();
        serde_json::Value::Array(calls)
    }

    let text = response.content_text().unwrap_or_default().to_string();
    let tool_calls = response.tool_calls();
    let finish_reason = response.finish_reason.as_ref();
    let model = response.model.clone().unwrap_or_default();
    let id = response.id.clone().unwrap_or_else(|| "siumai".to_string());

    match target {
        TargetJsonFormat::OpenAiChatCompletions => {
            let tool_calls_value = tool_calls_json(&tool_calls);
            let has_tool_calls =
                matches!(tool_calls_value, serde_json::Value::Array(ref v) if !v.is_empty());

            let mut message = serde_json::Map::new();
            message.insert(
                "role".to_string(),
                serde_json::Value::String("assistant".to_string()),
            );
            if text.trim().is_empty() && has_tool_calls {
                message.insert("content".to_string(), serde_json::Value::Null);
            } else {
                message.insert("content".to_string(), serde_json::Value::String(text));
            }
            if has_tool_calls {
                message.insert("tool_calls".to_string(), tool_calls_value);
            }

            let mut top = serde_json::Map::new();
            top.insert("id".to_string(), serde_json::Value::String(id));
            top.insert(
                "object".to_string(),
                serde_json::Value::String("chat.completion".to_string()),
            );
            top.insert(
                "created".to_string(),
                serde_json::Value::Number(serde_json::Number::from(0)),
            );
            top.insert("model".to_string(), serde_json::Value::String(model));
            top.insert(
                "choices".to_string(),
                serde_json::json!([{
                    "index": 0,
                    "message": serde_json::Value::Object(message),
                    "finish_reason": openai_finish_reason(finish_reason),
                }]),
            );
            if let Some(usage) = response.usage.as_ref() {
                top.insert("usage".to_string(), usage_json(usage));
            }
            if let Some(fp) = response.system_fingerprint.as_ref() {
                top.insert(
                    "system_fingerprint".to_string(),
                    serde_json::Value::String(fp.clone()),
                );
            }
            if let Some(tier) = response.service_tier.as_ref() {
                top.insert(
                    "service_tier".to_string(),
                    serde_json::Value::String(tier.clone()),
                );
            }
            serde_json::Value::Object(top)
        }
        TargetJsonFormat::OpenAiResponses => {
            let mut output: Vec<serde_json::Value> = Vec::new();

            if !text.trim().is_empty() {
                output.push(serde_json::json!({
                    "id": format!("msg_{id}"),
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        { "type": "output_text", "text": text }
                    ],
                }));
            }

            for call in tool_calls {
                if let ContentPart::ToolCall {
                    tool_call_id,
                    tool_name,
                    arguments,
                    ..
                } = call
                {
                    let args_str =
                        serde_json::to_string(arguments).unwrap_or_else(|_| arguments.to_string());
                    output.push(serde_json::json!({
                        "id": format!("fc_{tool_call_id}"),
                        "type": "function_call",
                        "call_id": tool_call_id,
                        "name": tool_name,
                        "arguments": args_str,
                    }));
                }
            }

            let mut top = serde_json::Map::new();
            top.insert("id".to_string(), serde_json::Value::String(id));
            top.insert(
                "object".to_string(),
                serde_json::Value::String("response".to_string()),
            );
            top.insert(
                "created".to_string(),
                serde_json::Value::Number(serde_json::Number::from(0)),
            );
            top.insert("model".to_string(), serde_json::Value::String(model));
            top.insert(
                "status".to_string(),
                serde_json::Value::String("completed".to_string()),
            );
            top.insert("output".to_string(), serde_json::Value::Array(output));
            top.insert(
                "output_text".to_string(),
                serde_json::Value::String(response.content.all_text()),
            );
            if let Some(usage) = response.usage.as_ref() {
                top.insert("usage".to_string(), usage_json(usage));
            }
            serde_json::Value::Object(top)
        }
        TargetJsonFormat::AnthropicMessages => {
            let mut content: Vec<serde_json::Value> = Vec::new();
            if !text.trim().is_empty() {
                content.push(serde_json::json!({ "type": "text", "text": text }));
            }
            for call in tool_calls {
                if let ContentPart::ToolCall {
                    tool_call_id,
                    tool_name,
                    arguments,
                    ..
                } = call
                {
                    content.push(serde_json::json!({
                        "type": "tool_use",
                        "id": tool_call_id,
                        "name": tool_name,
                        "input": arguments,
                    }));
                }
            }

            let usage = response.usage.as_ref().map(|u| {
                serde_json::json!({
                    "input_tokens": u.prompt_tokens,
                    "output_tokens": u.completion_tokens,
                })
            });

            serde_json::json!({
                "id": id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": content,
                "stop_reason": anthropic_stop_reason(finish_reason),
                "stop_sequence": serde_json::Value::Null,
                "usage": usage.unwrap_or(serde_json::json!({"input_tokens": 0, "output_tokens": 0})),
            })
        }
        TargetJsonFormat::GeminiGenerateContent => {
            let mut parts: Vec<serde_json::Value> = Vec::new();
            if !text.trim().is_empty() {
                parts.push(serde_json::json!({ "text": text }));
            }
            for call in tool_calls {
                if let ContentPart::ToolCall {
                    tool_name,
                    arguments,
                    ..
                } = call
                {
                    parts.push(serde_json::json!({
                        "functionCall": {
                            "name": tool_name,
                            "args": arguments,
                        }
                    }));
                }
            }

            let usage = response.usage.as_ref().map(|u| {
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
                serde_json::json!({
                    "promptTokenCount": u.prompt_tokens,
                    "candidatesTokenCount": u.completion_tokens,
                    "totalTokenCount": u.total_tokens,
                    "thoughtsTokenCount": thoughts,
                })
            });

            serde_json::json!({
                "candidates": [{
                    "content": { "role": "model", "parts": parts },
                    "finishReason": gemini_finish_reason(finish_reason),
                }],
                "usageMetadata": usage.unwrap_or(serde_json::Value::Null),
            })
        }
    }
}

/// Convert a unified `ChatResponse` into an Axum JSON response for the selected provider format.
pub fn to_transcoded_json_response(
    response: ChatResponse,
    target: TargetJsonFormat,
    opts: TranscodeJsonOptions,
) -> Response<Body> {
    let json = transcode_chat_response_to_json(&response, target);
    let body = if opts.pretty {
        serde_json::to_vec_pretty(&json)
    } else {
        serde_json::to_vec(&json)
    };

    match body {
        Ok(bytes) => {
            let mut resp = Response::new(Body::from(bytes));
            resp.headers_mut().insert(
                header::CONTENT_TYPE,
                header::HeaderValue::from_static("application/json"),
            );
            resp
        }
        Err(_) => Response::builder()
            .status(500)
            .header("content-type", "text/plain")
            .body(Body::from("failed to serialize response"))
            .unwrap_or_else(|_| Response::new(Body::from("internal error"))),
    }
}

/// Convert a unified `ChatResponse` into an Axum JSON response for the selected provider format,
/// with a caller-provided post-processing hook.
pub fn to_transcoded_json_response_with_transform<F>(
    response: ChatResponse,
    target: TargetJsonFormat,
    opts: TranscodeJsonOptions,
    transform: F,
) -> Response<Body>
where
    F: FnOnce(serde_json::Value) -> serde_json::Value + Send + Sync + 'static,
{
    let json = transcode_chat_response_to_json(&response, target);
    let json = transform(json);
    let body = if opts.pretty {
        serde_json::to_vec_pretty(&json)
    } else {
        serde_json::to_vec(&json)
    };

    match body {
        Ok(bytes) => {
            let mut resp = Response::new(Body::from(bytes));
            resp.headers_mut().insert(
                header::CONTENT_TYPE,
                header::HeaderValue::from_static("application/json"),
            );
            resp
        }
        Err(_) => Response::builder()
            .status(500)
            .header("content-type", "text/plain")
            .body(Body::from("failed to serialize response"))
            .unwrap_or_else(|_| Response::new(Body::from("internal error"))),
    }
}

#[cfg(feature = "openai")]
fn openai_responses_supports_v3_part_type(tpe: &str) -> bool {
    matches!(
        tpe,
        "stream-start"
            | "response-metadata"
            | "text-start"
            | "text-delta"
            | "text-end"
            | "reasoning-start"
            | "reasoning-delta"
            | "reasoning-end"
            | "tool-input-start"
            | "tool-input-delta"
            | "tool-input-end"
            | "tool-approval-request"
            | "tool-call"
            | "tool-result"
            | "source"
            | "finish"
            | "error"
    )
}

#[cfg(feature = "openai")]
fn apply_openai_responses_v3_policy(
    stream: ChatStream,
    behavior: siumai::experimental::streaming::V3UnsupportedPartBehavior,
) -> ChatStream {
    use siumai::experimental::streaming::{LanguageModelV3StreamPart, transform_chat_event_stream};

    transform_chat_event_stream(stream, move |ev| match ev {
        ChatStreamEvent::Custom { event_type, data } => {
            let Some(tpe) = data.get("type").and_then(|v| v.as_str()) else {
                return vec![ChatStreamEvent::Custom { event_type, data }];
            };

            if openai_responses_supports_v3_part_type(tpe) {
                return vec![ChatStreamEvent::Custom { event_type, data }];
            }

            let Some(part) = LanguageModelV3StreamPart::parse_loose_json(&data) else {
                // Unknown shape; keep it and let the downstream serializer decide.
                return vec![ChatStreamEvent::Custom { event_type, data }];
            };

            match behavior {
                siumai::experimental::streaming::V3UnsupportedPartBehavior::Drop => Vec::new(),
                siumai::experimental::streaming::V3UnsupportedPartBehavior::AsText => part
                    .to_lossy_text()
                    .map(|text| {
                        vec![ChatStreamEvent::ContentDelta {
                            delta: text,
                            index: None,
                        }]
                    })
                    .unwrap_or_default(),
            }
        }
        other => vec![other],
    })
}

/// Options for SSE encoding.
///
/// Controls which events are included in the SSE stream and how errors are handled.
#[derive(Debug, Clone)]
pub struct SseOptions {
    /// Whether to include usage updates frames.
    ///
    /// When `true`, emits `event: usage` with token usage information.
    /// Default: `true`
    pub include_usage: bool,

    /// Whether to include a final `end` event with the full response JSON.
    ///
    /// When `true`, emits `event: end` with the complete response.
    /// Default: `true`
    pub include_end: bool,

    /// Whether to include the initial `start` event with metadata.
    ///
    /// When `true`, emits `event: start` at the beginning of the stream.
    /// Default: `true`
    pub include_start: bool,

    /// Whether to mask error messages for security.
    ///
    /// When `true`, replaces detailed error messages with "internal error".
    /// Recommended for production environments to avoid leaking sensitive information.
    /// Default: `true`
    pub mask_errors: bool,

    /// Custom error message to use when `mask_errors` is `true`.
    ///
    /// If `None`, uses "internal error" as the default masked message.
    /// Default: `None`
    pub masked_error_message: Option<String>,
}

impl Default for SseOptions {
    fn default() -> Self {
        Self {
            include_usage: true,
            include_end: true,
            include_start: true,
            mask_errors: true,
            masked_error_message: None,
        }
    }
}

impl SseOptions {
    /// Create options suitable for development (errors not masked).
    pub fn development() -> Self {
        Self {
            mask_errors: false,
            ..Default::default()
        }
    }

    /// Create options suitable for production (errors masked).
    pub fn production() -> Self {
        Self {
            mask_errors: true,
            ..Default::default()
        }
    }

    /// Create minimal options (only content deltas, no metadata).
    pub fn minimal() -> Self {
        Self {
            include_usage: false,
            include_end: false,
            include_start: false,
            mask_errors: true,
            masked_error_message: None,
        }
    }
}

/// Convert a `ChatStream` into an Axum SSE response.
///
/// This function converts the stream into a format compatible with Axum's `Sse` type,
/// automatically handling errors and converting them to SSE events.
///
/// ## Arguments
///
/// - `stream`: The `ChatStream` to convert
/// - `opts`: SSE encoding options (use `SseOptions::production()` for production)
///
/// ## Returns
///
/// An `Sse<impl Stream>` that can be returned directly from an Axum handler.
///
/// ## Example
///
/// ```rust,no_run
/// use axum::response::sse::Sse;
/// use siumai_extras::server::axum::{to_sse_response, SseOptions};
/// use siumai::prelude::unified::ChatStream;
///
/// async fn handler(stream: ChatStream) -> Sse<impl futures::Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>>> {
///     to_sse_response(stream, SseOptions::production())
/// }
/// ```
pub fn to_sse_response(
    stream: ChatStream,
    opts: SseOptions,
) -> Sse<impl Stream<Item = Result<Event, Infallible>> + Send> {
    let event_stream = stream.map(move |item| {
        let event = match item {
            Ok(ChatStreamEvent::StreamStart { metadata }) => {
                if opts.include_start {
                    let data =
                        serde_json::to_string(&metadata).unwrap_or_else(|_| "{}".to_string());
                    Some(Event::default().event("start").data(data))
                } else {
                    None
                }
            }
            Ok(ChatStreamEvent::ContentDelta { delta, index }) => {
                let data = serde_json::json!({"delta": delta, "index": index});
                let data_str = serde_json::to_string(&data).unwrap_or_else(|_| "{}".to_string());
                Some(Event::default().event("delta").data(data_str))
            }
            Ok(ChatStreamEvent::ToolCallDelta {
                id,
                function_name,
                arguments_delta,
                index,
            }) => {
                let data = serde_json::json!({
                    "id": id,
                    "name": function_name,
                    "arguments_delta": arguments_delta,
                    "index": index
                });
                let data_str = serde_json::to_string(&data).unwrap_or_else(|_| "{}".to_string());
                Some(Event::default().event("tool").data(data_str))
            }
            Ok(ChatStreamEvent::ThinkingDelta { delta }) => {
                let data = serde_json::json!({"delta": delta});
                let data_str = serde_json::to_string(&data).unwrap_or_else(|_| "{}".to_string());
                Some(Event::default().event("reasoning").data(data_str))
            }
            Ok(ChatStreamEvent::UsageUpdate { usage }) => {
                if opts.include_usage {
                    let data = serde_json::to_string(&usage).unwrap_or_else(|_| "{}".to_string());
                    Some(Event::default().event("usage").data(data))
                } else {
                    None
                }
            }
            Ok(ChatStreamEvent::StreamEnd { response }) => {
                if opts.include_end {
                    let data =
                        serde_json::to_string(&response).unwrap_or_else(|_| "{}".to_string());
                    Some(Event::default().event("end").data(data))
                } else {
                    None
                }
            }
            Ok(ChatStreamEvent::Error { error }) | Err(LlmError::InternalError(error)) => {
                let msg = if opts.mask_errors {
                    opts.masked_error_message
                        .clone()
                        .unwrap_or_else(|| "internal error".to_string())
                } else {
                    error
                };
                let data = serde_json::json!({"error": msg});
                let data_str = serde_json::to_string(&data)
                    .unwrap_or_else(|_| r#"{"error":"internal error"}"#.to_string());
                Some(Event::default().event("error").data(data_str))
            }
            Ok(ChatStreamEvent::Custom { event_type, data }) => {
                // Forward custom events as-is
                let data_str = serde_json::to_string(&data).unwrap_or_else(|_| "{}".to_string());
                Some(Event::default().event(&event_type).data(data_str))
            }
            Err(e) => {
                let msg = if opts.mask_errors {
                    opts.masked_error_message
                        .clone()
                        .unwrap_or_else(|| "internal error".to_string())
                } else {
                    e.user_message()
                };
                let data = serde_json::json!({"error": msg});
                let data_str = serde_json::to_string(&data)
                    .unwrap_or_else(|_| r#"{"error":"internal error"}"#.to_string());
                Some(Event::default().event("error").data(data_str))
            }
        };

        // Convert Option<Event> to Result<Event, Infallible>
        // If None, we skip this event by returning an empty event
        Ok(event.unwrap_or_else(|| Event::default().comment("skipped")))
    });

    Sse::new(event_stream)
}

/// Convert a `ChatStream` into a plain text stream for Axum.
///
/// This function extracts only the content deltas from the stream,
/// ignoring all other events (usage, metadata, etc.).
///
/// ## Arguments
///
/// - `stream`: The `ChatStream` to convert
///
/// ## Returns
///
/// A pinned stream of text chunks that can be used with Axum's streaming response.
///
/// ## Example
///
/// ```rust,no_run
/// use axum::response::Response;
/// use axum::body::Body;
/// use siumai_extras::server::axum::to_text_stream;
/// use siumai::prelude::unified::ChatStream;
///
/// async fn handler(stream: ChatStream) -> Response<Body> {
///     let text_stream = to_text_stream(stream);
///     Response::new(Body::from_stream(text_stream))
/// }
/// ```
pub fn to_text_stream(
    stream: ChatStream,
) -> Pin<Box<dyn Stream<Item = Result<String, Infallible>> + Send>> {
    let text_stream = stream.filter_map(|item| async move {
        match item {
            Ok(ChatStreamEvent::ContentDelta { delta, .. }) => Some(Ok(delta)),
            Ok(ChatStreamEvent::Error { error }) => {
                // For text streams, we can include error as text
                Some(Ok(format!("\n[Error: {}]\n", error)))
            }
            Err(e) => {
                // Convert LlmError to text
                Some(Ok(format!("\n[Error: {}]\n", e.user_message())))
            }
            _ => None, // Skip other events
        }
    });

    Box::pin(text_stream)
}

/// Convert a `ChatStream` into an OpenAI Responses-compatible SSE byte stream.
///
/// This is a convenience helper for building gateways/proxies:
/// - It bridges provider-specific Vercel-aligned stream parts (e.g. `gemini:*`, `anthropic:*`)
///   into `openai:*` stream parts.
/// - It then serializes the unified stream into OpenAI Responses SSE frames.
/// - It never yields stream-level errors (errors become `response.error` SSE frames).
///
/// Note: requires the `openai` feature (and the OpenAI Responses streaming implementation).
#[cfg(feature = "openai")]
pub fn to_openai_responses_sse_stream(
    stream: ChatStream,
) -> Pin<Box<dyn Stream<Item = Result<axum::body::Bytes, Infallible>> + Send>> {
    to_openai_responses_sse_stream_with_options(stream, TranscodeSseOptions::default())
}

/// Convert a `ChatStream` into an OpenAI Responses-compatible SSE byte stream with configurable
/// v3 fallback options.
#[cfg(feature = "openai")]
pub fn to_openai_responses_sse_stream_with_options(
    stream: ChatStream,
    opts: TranscodeSseOptions,
) -> Pin<Box<dyn Stream<Item = Result<axum::body::Bytes, Infallible>> + Send>> {
    use siumai::experimental::streaming::{
        OpenAiResponsesStreamPartsBridge, encode_chat_stream_as_sse, transform_chat_event_stream,
    };
    use siumai::protocol::openai::responses_sse::OpenAiResponsesEventConverter;

    let mut bridge = OpenAiResponsesStreamPartsBridge::new();

    let stream = apply_openai_responses_v3_policy(stream, opts.v3_unsupported_part_behavior);
    let bridged = transform_chat_event_stream(stream, move |ev| bridge.bridge_event(ev));

    let converter = OpenAiResponsesEventConverter::new();
    let bytes_stream = encode_chat_stream_as_sse(bridged, converter);

    Box::pin(bytes_stream.map(|item| match item {
        Ok(bytes) => Ok(bytes),
        Err(e) => Ok(axum::body::Bytes::from(format!(
            "event: response.error\ndata: {{\"type\":\"response.error\",\"error\":{{\"message\":{}}}}}\n\n",
            serde_json::json!(e.user_message())
        ))),
    }))
}

/// Convert a `ChatStream` into an Axum `Response<Body>` in OpenAI Responses SSE format.
///
/// This is a thin wrapper around `to_openai_responses_sse_stream()` that sets the proper
/// `Content-Type: text/event-stream` header.
#[cfg(feature = "openai")]
pub fn to_openai_responses_sse_response(stream: ChatStream) -> Response<Body> {
    to_openai_responses_sse_response_with_options(stream, TranscodeSseOptions::default())
}

/// Convert a `ChatStream` into an Axum `Response<Body>` in OpenAI Responses SSE format with
/// configurable v3 fallback options.
#[cfg(feature = "openai")]
pub fn to_openai_responses_sse_response_with_options(
    stream: ChatStream,
    opts: TranscodeSseOptions,
) -> Response<Body> {
    let body = Body::from_stream(to_openai_responses_sse_stream_with_options(stream, opts));
    let mut resp = Response::new(body);
    resp.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("text/event-stream"),
    );
    resp
}

/// Convert a `ChatStream` into an OpenAI-compatible Chat Completions SSE response.
///
/// This is useful when building gateways that need to expose an OpenAI-compatible endpoint
/// backed by a non-OpenAI provider.
#[cfg(feature = "openai")]
pub fn to_openai_chat_completions_sse_response(stream: ChatStream) -> Response<Body> {
    to_openai_chat_completions_sse_response_with_options(stream, TranscodeSseOptions::default())
}

/// Convert a `ChatStream` into an OpenAI-compatible Chat Completions SSE response with configurable
/// v3 fallback options.
#[cfg(feature = "openai")]
pub fn to_openai_chat_completions_sse_response_with_options(
    stream: ChatStream,
    opts: TranscodeSseOptions,
) -> Response<Body> {
    use siumai::experimental::streaming::encode_chat_stream_as_sse;
    use siumai::protocol::openai::compat::openai_config::OpenAiCompatibleConfig;
    use siumai::protocol::openai::compat::provider_registry::{
        ConfigurableAdapter, ProviderConfig, ProviderFieldMappings,
    };
    use siumai::protocol::openai::compat::streaming::OpenAiCompatibleEventConverter;

    let adapter = std::sync::Arc::new(ConfigurableAdapter::new(ProviderConfig {
        id: "openai".to_string(),
        name: "OpenAI".to_string(),
        base_url: "https://api.openai.com/v1".to_string(),
        field_mappings: ProviderFieldMappings::default(),
        capabilities: vec!["tools".to_string()],
        default_model: Some("gpt-4o-mini".to_string()),
        supports_reasoning: false,
        api_key_env: None,
        api_key_env_aliases: vec![],
    }));

    let cfg = OpenAiCompatibleConfig::new(
        "openai",
        "sk-siumai-encoding-only",
        "https://api.openai.com/v1",
        adapter.clone(),
    )
    .with_model("gpt-4o-mini");

    let converter = OpenAiCompatibleEventConverter::new(cfg, adapter)
        .with_v3_unsupported_part_behavior(opts.v3_unsupported_part_behavior);

    let bytes = encode_chat_stream_as_sse(stream, converter);
    let body = Body::from_stream(bytes.map(|item| {
        Ok::<axum::body::Bytes, Infallible>(match item {
            Ok(bytes) => bytes,
            Err(e) => axum::body::Bytes::from(format!(
                "event: error\ndata: {}\n\n",
                serde_json::json!({ "error": e.user_message() })
            )),
        })
    }));

    let mut resp = Response::new(body);
    resp.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("text/event-stream"),
    );
    resp
}

/// Convert a `ChatStream` into an Anthropic Messages SSE response (best-effort).
#[cfg(feature = "anthropic")]
pub fn to_anthropic_messages_sse_response(stream: ChatStream) -> Response<Body> {
    use futures::StreamExt;
    use siumai::experimental::streaming::encode_chat_stream_as_sse;
    use siumai::protocol::anthropic::params::AnthropicParams;
    use siumai::protocol::anthropic::streaming::AnthropicEventConverter;

    let converter = AnthropicEventConverter::new(AnthropicParams::default());
    let bytes = encode_chat_stream_as_sse(stream, converter);

    let body = Body::from_stream(bytes.map(|item| {
        Ok::<axum::body::Bytes, Infallible>(match item {
            Ok(bytes) => bytes,
            Err(e) => axum::body::Bytes::from(format!(
                "data: {}\n\n",
                serde_json::json!({
                    "type": "error",
                    "error": { "type": "api_error", "message": e.user_message() }
                })
            )),
        })
    }));

    let mut resp = Response::new(body);
    resp.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("text/event-stream"),
    );
    resp
}

/// Convert a `ChatStream` into a Google Gemini GenerateContent SSE response (best-effort).
#[cfg(feature = "google")]
pub fn to_gemini_generate_content_sse_response(stream: ChatStream) -> Response<Body> {
    use futures::StreamExt;
    use siumai::experimental::streaming::encode_chat_stream_as_sse;
    use siumai::protocol::gemini::streaming::GeminiEventConverter;
    use siumai::protocol::gemini::types::GeminiConfig;

    let cfg = GeminiConfig::default();
    let converter = GeminiEventConverter::new(cfg);
    let bytes = encode_chat_stream_as_sse(stream, converter);

    let body = Body::from_stream(bytes.map(|item| {
        Ok::<axum::body::Bytes, Infallible>(match item {
            Ok(bytes) => bytes,
            Err(e) => axum::body::Bytes::from(format!(
                "data: {}\n\n",
                serde_json::json!({
                    "error": { "message": e.user_message() }
                })
            )),
        })
    }));

    let mut resp = Response::new(body);
    resp.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("text/event-stream"),
    );
    resp
}

/// Convert a `ChatStream` into a provider-native SSE response with a unified target selector.
///
/// This helper is meant for gateways/proxies that want to expose multiple protocol surfaces
/// backed by the same upstream stream.
pub fn to_transcoded_sse_response(
    stream: ChatStream,
    target: TargetSseFormat,
    opts: TranscodeSseOptions,
) -> Response<Body> {
    match target {
        TargetSseFormat::OpenAiResponses => {
            #[cfg(feature = "openai")]
            {
                if opts.bridge_openai_responses_stream_parts {
                    return to_openai_responses_sse_response_with_options(stream, opts);
                }
                let stream =
                    apply_openai_responses_v3_policy(stream, opts.v3_unsupported_part_behavior);
                // Without bridge: serialize directly (best-effort).
                use futures::StreamExt;
                use siumai::experimental::streaming::encode_chat_stream_as_sse;
                use siumai::protocol::openai::responses_sse::OpenAiResponsesEventConverter;
                let bytes = encode_chat_stream_as_sse(stream, OpenAiResponsesEventConverter::new());
                let body = Body::from_stream(bytes.map(|item| {
                    Ok::<axum::body::Bytes, Infallible>(match item {
                        Ok(bytes) => bytes,
                        Err(e) => axum::body::Bytes::from(format!(
                            "event: response.error\ndata: {{\"type\":\"response.error\",\"error\":{{\"message\":{}}}}}\n\n",
                            serde_json::json!(e.user_message())
                        )),
                    })
                }));
                let mut resp = Response::new(body);
                resp.headers_mut().insert(
                    header::CONTENT_TYPE,
                    header::HeaderValue::from_static("text/event-stream"),
                );
                resp
            }
            #[cfg(not(feature = "openai"))]
            {
                let _ = opts;
                let _ = stream;
                Response::builder()
                    .status(501)
                    .body(Body::from("openai feature is disabled"))
                    .unwrap()
            }
        }
        TargetSseFormat::OpenAiChatCompletions => {
            #[cfg(feature = "openai")]
            {
                to_openai_chat_completions_sse_response_with_options(stream, opts)
            }
            #[cfg(not(feature = "openai"))]
            {
                let _ = opts;
                let _ = stream;
                Response::builder()
                    .status(501)
                    .body(Body::from("openai feature is disabled"))
                    .unwrap()
            }
        }
        TargetSseFormat::AnthropicMessages => {
            #[cfg(feature = "anthropic")]
            {
                use futures::StreamExt;
                use siumai::experimental::streaming::encode_chat_stream_as_sse;
                use siumai::protocol::anthropic::params::AnthropicParams;
                use siumai::protocol::anthropic::streaming::AnthropicEventConverter;

                let converter = AnthropicEventConverter::new(AnthropicParams::default())
                    .with_v3_unsupported_part_behavior(opts.v3_unsupported_part_behavior);
                let bytes = encode_chat_stream_as_sse(stream, converter);

                let body = Body::from_stream(bytes.map(|item| {
                    Ok::<axum::body::Bytes, Infallible>(match item {
                        Ok(bytes) => bytes,
                        Err(e) => axum::body::Bytes::from(format!(
                            "data: {}\n\n",
                            serde_json::json!({
                                "type": "error",
                                "error": { "type": "api_error", "message": e.user_message() }
                            })
                        )),
                    })
                }));

                let mut resp = Response::new(body);
                resp.headers_mut().insert(
                    header::CONTENT_TYPE,
                    header::HeaderValue::from_static("text/event-stream"),
                );
                resp
            }
            #[cfg(not(feature = "anthropic"))]
            {
                let _ = opts;
                let _ = stream;
                Response::builder()
                    .status(501)
                    .body(Body::from("anthropic feature is disabled"))
                    .unwrap()
            }
        }
        TargetSseFormat::GeminiGenerateContent => {
            #[cfg(feature = "google")]
            {
                use futures::StreamExt;
                use siumai::experimental::streaming::encode_chat_stream_as_sse;
                use siumai::protocol::gemini::streaming::GeminiEventConverter;
                use siumai::protocol::gemini::types::GeminiConfig;

                let converter = GeminiEventConverter::new(GeminiConfig::default())
                    .with_v3_unsupported_part_behavior(opts.v3_unsupported_part_behavior)
                    .with_emit_function_response_tool_results(
                        opts.gemini_emit_function_response_tool_results,
                    );
                let bytes = encode_chat_stream_as_sse(stream, converter);

                let body = Body::from_stream(bytes.map(|item| {
                    Ok::<axum::body::Bytes, Infallible>(match item {
                        Ok(bytes) => bytes,
                        Err(e) => axum::body::Bytes::from(format!(
                            "data: {}\n\n",
                            serde_json::json!({
                                "error": { "message": e.user_message() }
                            })
                        )),
                    })
                }));

                let mut resp = Response::new(body);
                resp.headers_mut().insert(
                    header::CONTENT_TYPE,
                    header::HeaderValue::from_static("text/event-stream"),
                );
                resp
            }
            #[cfg(not(feature = "google"))]
            {
                let _ = opts;
                let _ = stream;
                Response::builder()
                    .status(501)
                    .body(Body::from("google feature is disabled"))
                    .unwrap()
            }
        }
    }
}

/// Convert a `ChatStream` into a provider-native SSE response with a unified target selector,
/// with a caller-provided event transform hook.
///
/// The `transform` closure is applied to each `ChatStreamEvent` (expanding or dropping events)
/// before the target-specific transcoding/bridging logic runs.
pub fn to_transcoded_sse_response_with_transform<F>(
    stream: ChatStream,
    target: TargetSseFormat,
    opts: TranscodeSseOptions,
    transform: F,
) -> Response<Body>
where
    F: FnMut(ChatStreamEvent) -> Vec<ChatStreamEvent> + Send + Sync + 'static,
{
    use siumai::experimental::streaming::transform_chat_event_stream;

    let stream = transform_chat_event_stream(stream, transform);
    to_transcoded_sse_response(stream, target, opts)
}

#[cfg(test)]
mod transcode_tests {
    use super::*;

    #[test]
    fn transcode_options_defaults_are_stable() {
        let opts = TranscodeSseOptions::default();
        assert_eq!(
            opts.v3_unsupported_part_behavior,
            siumai::experimental::streaming::V3UnsupportedPartBehavior::Drop
        );
        assert!(opts.bridge_openai_responses_stream_parts);
    }

    #[tokio::test]
    async fn to_transcoded_sse_response_with_transform_builds() {
        use futures::stream;
        use siumai::prelude::unified::{ChatResponse, ChatStreamEvent, MessageContent};

        let chat_stream: ChatStream = Box::pin(stream::iter(vec![
            Ok(ChatStreamEvent::ContentDelta {
                delta: "hello".to_string(),
                index: None,
            }),
            Ok(ChatStreamEvent::StreamEnd {
                response: ChatResponse::new(MessageContent::Text("done".to_string())),
            }),
        ]));

        let _resp = to_transcoded_sse_response_with_transform(
            chat_stream,
            TargetSseFormat::OpenAiResponses,
            TranscodeSseOptions::default(),
            |ev| vec![ev],
        );
    }

    #[tokio::test]
    #[cfg(all(feature = "openai", feature = "anthropic", feature = "google"))]
    async fn v3_raw_and_file_parts_follow_v3_unsupported_part_behavior_across_targets() {
        use futures::StreamExt;
        use siumai::experimental::streaming::{ChatByteStream, encode_chat_stream_as_sse};
        use siumai::prelude::unified::{ChatResponse, MessageContent};

        fn test_stream() -> ChatStream {
            use futures::stream;
            use siumai::prelude::unified::{ChatStreamEvent, ResponseMetadata};

            let events = vec![
                Ok(ChatStreamEvent::StreamStart {
                    metadata: ResponseMetadata {
                        id: Some("test-id".to_string()),
                        model: Some("test-model".to_string()),
                        created: None,
                        provider: "test".to_string(),
                        request_id: None,
                    },
                }),
                Ok(ChatStreamEvent::Custom {
                    event_type: "bridge:test".to_string(),
                    data: serde_json::json!({
                        "type": "raw",
                        "rawValue": { "hello": "world" }
                    }),
                }),
                Ok(ChatStreamEvent::Custom {
                    event_type: "bridge:test".to_string(),
                    data: serde_json::json!({
                        "type": "file",
                        "mediaType": "text/plain",
                        "data": "aGVsbG8="
                    }),
                }),
                Ok(ChatStreamEvent::StreamEnd {
                    response: ChatResponse::new(MessageContent::Text("done".to_string())),
                }),
            ];

            Box::pin(stream::iter(events))
        }

        async fn collect_bytes(mut s: ChatByteStream) -> Vec<u8> {
            let mut out = Vec::new();
            while let Some(item) = s.next().await {
                let chunk = item.expect("encode ok");
                out.extend_from_slice(chunk.as_ref());
            }
            out
        }

        async fn collect_infallible_bytes(
            mut s: Pin<Box<dyn Stream<Item = Result<axum::body::Bytes, Infallible>> + Send>>,
        ) -> Vec<u8> {
            let mut out = Vec::new();
            while let Some(item) = s.next().await {
                let chunk = item.expect("encode ok");
                out.extend_from_slice(chunk.as_ref());
            }
            out
        }

        // OpenAI Responses
        {
            let strict_opts = TranscodeSseOptions {
                v3_unsupported_part_behavior:
                    siumai::experimental::streaming::V3UnsupportedPartBehavior::Drop,
                ..TranscodeSseOptions::default()
            };
            let lossy_opts = TranscodeSseOptions {
                v3_unsupported_part_behavior:
                    siumai::experimental::streaming::V3UnsupportedPartBehavior::AsText,
                ..TranscodeSseOptions::default()
            };

            let strict = collect_infallible_bytes(to_openai_responses_sse_stream_with_options(
                test_stream(),
                strict_opts,
            ))
            .await;
            let strict_text = String::from_utf8_lossy(&strict);
            assert!(!strict_text.contains("[raw]"));
            assert!(!strict_text.contains("[file]"));

            let lossy = collect_infallible_bytes(to_openai_responses_sse_stream_with_options(
                test_stream(),
                lossy_opts,
            ))
            .await;
            let lossy_text = String::from_utf8_lossy(&lossy);
            assert!(lossy_text.contains("[raw]"));
            assert!(lossy_text.contains("[file]"));
        }

        // OpenAI Chat Completions
        {
            use siumai::protocol::openai::compat::openai_config::OpenAiCompatibleConfig;
            use siumai::protocol::openai::compat::provider_registry::{
                ConfigurableAdapter, ProviderConfig, ProviderFieldMappings,
            };
            use siumai::protocol::openai::compat::streaming::OpenAiCompatibleEventConverter;
            use std::sync::Arc;

            let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
                id: "openai".to_string(),
                name: "OpenAI".to_string(),
                base_url: "https://api.openai.com/v1".to_string(),
                field_mappings: ProviderFieldMappings::default(),
                capabilities: vec!["tools".to_string()],
                default_model: Some("gpt-4o-mini".to_string()),
                supports_reasoning: false,
                api_key_env: None,
                api_key_env_aliases: vec![],
            }));

            let cfg = OpenAiCompatibleConfig::new(
                "openai",
                "sk-siumai-encoding-only",
                "https://api.openai.com/v1",
                adapter.clone(),
            )
            .with_model("gpt-4o-mini");

            let strict_conv = OpenAiCompatibleEventConverter::new(cfg.clone(), adapter.clone())
                .with_v3_unsupported_part_behavior(
                    siumai::experimental::streaming::V3UnsupportedPartBehavior::Drop,
                );
            let lossy_conv = OpenAiCompatibleEventConverter::new(cfg, adapter)
                .with_v3_unsupported_part_behavior(
                    siumai::experimental::streaming::V3UnsupportedPartBehavior::AsText,
                );

            let strict = collect_bytes(encode_chat_stream_as_sse(test_stream(), strict_conv)).await;
            let strict_text = String::from_utf8_lossy(&strict);
            assert!(!strict_text.contains("[raw]"));
            assert!(!strict_text.contains("[file]"));

            let lossy = collect_bytes(encode_chat_stream_as_sse(test_stream(), lossy_conv)).await;
            let lossy_text = String::from_utf8_lossy(&lossy);
            assert!(lossy_text.contains("[raw]"));
            assert!(lossy_text.contains("[file]"));
        }

        // Anthropic Messages
        {
            use siumai::protocol::anthropic::params::AnthropicParams;
            use siumai::protocol::anthropic::streaming::AnthropicEventConverter;

            let strict_conv = AnthropicEventConverter::new(AnthropicParams::default())
                .with_v3_unsupported_part_behavior(
                    siumai::experimental::streaming::V3UnsupportedPartBehavior::Drop,
                );
            let lossy_conv = AnthropicEventConverter::new(AnthropicParams::default())
                .with_v3_unsupported_part_behavior(
                    siumai::experimental::streaming::V3UnsupportedPartBehavior::AsText,
                );

            let strict = collect_bytes(encode_chat_stream_as_sse(test_stream(), strict_conv)).await;
            let strict_text = String::from_utf8_lossy(&strict);
            assert!(!strict_text.contains("[raw]"));
            assert!(!strict_text.contains("[file]"));

            let lossy = collect_bytes(encode_chat_stream_as_sse(test_stream(), lossy_conv)).await;
            let lossy_text = String::from_utf8_lossy(&lossy);
            assert!(lossy_text.contains("[raw]"));
            assert!(lossy_text.contains("[file]"));
        }

        // Gemini GenerateContent
        {
            use siumai::protocol::gemini::streaming::GeminiEventConverter;
            use siumai::protocol::gemini::types::GeminiConfig;

            let strict_conv = GeminiEventConverter::new(GeminiConfig::default())
                .with_v3_unsupported_part_behavior(
                    siumai::experimental::streaming::V3UnsupportedPartBehavior::Drop,
                );
            let lossy_conv = GeminiEventConverter::new(GeminiConfig::default())
                .with_v3_unsupported_part_behavior(
                    siumai::experimental::streaming::V3UnsupportedPartBehavior::AsText,
                );

            let strict = collect_bytes(encode_chat_stream_as_sse(test_stream(), strict_conv)).await;
            let strict_text = String::from_utf8_lossy(&strict);
            assert!(!strict_text.contains("[raw]"));
            assert!(!strict_text.contains("[file]"));

            let lossy = collect_bytes(encode_chat_stream_as_sse(test_stream(), lossy_conv)).await;
            let lossy_text = String::from_utf8_lossy(&lossy);
            assert!(lossy_text.contains("[raw]"));
            assert!(lossy_text.contains("[file]"));
        }
    }
}

#[cfg(test)]
mod json_transcode_tests {
    use super::*;
    use serde_json::json;
    use siumai::prelude::unified::MessageContent;

    #[test]
    fn openai_chat_completions_json_includes_tool_calls() {
        let resp = ChatResponse::new(MessageContent::MultiModal(vec![
            ContentPart::text("ok"),
            ContentPart::tool_call("call_1", "get_weather", json!({"city":"GZ"}), None),
        ]));

        let v = transcode_chat_response_to_json(&resp, TargetJsonFormat::OpenAiChatCompletions);
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["message"]["role"], "assistant");
        assert_eq!(v["choices"][0]["message"]["tool_calls"][0]["id"], "call_1");
        assert_eq!(
            v["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
    }

    #[test]
    fn anthropic_messages_json_emits_tool_use_blocks() {
        let resp = ChatResponse::new(MessageContent::MultiModal(vec![
            ContentPart::text("thinking"),
            ContentPart::tool_call("call_1", "search", json!({"q":"rust"}), None),
        ]));

        let v = transcode_chat_response_to_json(&resp, TargetJsonFormat::AnthropicMessages);
        assert_eq!(v["type"], "message");
        assert_eq!(v["role"], "assistant");
        assert_eq!(v["content"][1]["type"], "tool_use");
        assert_eq!(v["content"][1]["id"], "call_1");
        assert_eq!(v["content"][1]["name"], "search");
    }

    #[test]
    fn gemini_generate_content_json_emits_function_call_parts() {
        let resp = ChatResponse::new(MessageContent::MultiModal(vec![
            ContentPart::text("ok"),
            ContentPart::tool_call("call_1", "search", json!({"q":"rust"}), None),
        ]));

        let v = transcode_chat_response_to_json(&resp, TargetJsonFormat::GeminiGenerateContent);
        assert_eq!(v["candidates"][0]["content"]["role"], "model");
        assert_eq!(
            v["candidates"][0]["content"]["parts"][1]["functionCall"]["name"],
            "search"
        );
    }

    #[test]
    fn openai_responses_json_includes_output_items() {
        let resp = ChatResponse::new(MessageContent::MultiModal(vec![
            ContentPart::text("ok"),
            ContentPart::tool_call("call_1", "search", json!({"q":"rust"}), None),
        ]));

        let v = transcode_chat_response_to_json(&resp, TargetJsonFormat::OpenAiResponses);
        assert_eq!(v["object"], "response");
        assert_eq!(v["status"], "completed");
        assert_eq!(v["output"][0]["type"], "message");
        assert_eq!(v["output"][1]["type"], "function_call");
        assert_eq!(v["output"][1]["call_id"], "call_1");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;
    use siumai::prelude::unified::{ChatResponse, MessageContent, ResponseMetadata, Usage};

    #[tokio::test]
    async fn test_to_sse_response_basic() {
        let events = vec![
            Ok(ChatStreamEvent::StreamStart {
                metadata: ResponseMetadata {
                    id: Some("test-id".to_string()),
                    model: Some("gpt-4".to_string()),
                    created: Some(chrono::Utc::now()),
                    provider: "openai".to_string(),
                    request_id: None,
                },
            }),
            Ok(ChatStreamEvent::ContentDelta {
                delta: "Hello".to_string(),
                index: Some(0),
            }),
            Ok(ChatStreamEvent::ContentDelta {
                delta: " world".to_string(),
                index: Some(0),
            }),
            Ok(ChatStreamEvent::StreamEnd {
                response: ChatResponse::new(MessageContent::Text("Hello world".to_string())),
            }),
        ];

        let chat_stream: ChatStream = Box::pin(stream::iter(events));
        let _sse = to_sse_response(chat_stream, SseOptions::default());
    }

    #[tokio::test]
    async fn test_to_text_stream_basic() {
        let events = vec![
            Ok(ChatStreamEvent::ContentDelta {
                delta: "Hello".to_string(),
                index: Some(0),
            }),
            Ok(ChatStreamEvent::UsageUpdate {
                usage: Usage::new(10, 5),
            }),
            Ok(ChatStreamEvent::ContentDelta {
                delta: " world".to_string(),
                index: Some(0),
            }),
        ];

        let chat_stream: ChatStream = Box::pin(stream::iter(events));
        let text_stream = to_text_stream(chat_stream);

        let collected: Vec<_> = text_stream.collect().await;
        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0].as_ref().unwrap(), "Hello");
        assert_eq!(collected[1].as_ref().unwrap(), " world");
    }

    #[tokio::test]
    #[cfg(feature = "openai")]
    async fn test_to_openai_chat_completions_sse_response_builds() {
        let events = vec![Ok(ChatStreamEvent::ContentDelta {
            delta: "Hello".to_string(),
            index: Some(0),
        })];
        let chat_stream: ChatStream = Box::pin(stream::iter(events));
        let _resp = to_openai_chat_completions_sse_response(chat_stream);
    }
}
