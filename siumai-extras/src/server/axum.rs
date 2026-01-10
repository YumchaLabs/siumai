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

use siumai::prelude::unified::{ChatStream, ChatStreamEvent, LlmError};

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
    use futures::stream;
    use siumai::experimental::streaming::{
        OpenAiResponsesStreamPartsBridge, encode_chat_stream_as_sse,
    };
    use siumai::protocol::openai::responses_sse::OpenAiResponsesEventConverter;

    let mut bridge = OpenAiResponsesStreamPartsBridge::new();

    let bridged = stream.flat_map(move |item| {
        let events: Vec<Result<ChatStreamEvent, LlmError>> = match item {
            Ok(ev) => bridge.bridge_event(ev).into_iter().map(Ok).collect(),
            Err(e) => vec![Ok(ChatStreamEvent::Error {
                error: e.user_message(),
            })],
        };
        stream::iter(events)
    });

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
    let body = Body::from_stream(to_openai_responses_sse_stream(stream));
    let mut resp = Response::new(body);
    resp.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("text/event-stream"),
    );
    resp
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
}
