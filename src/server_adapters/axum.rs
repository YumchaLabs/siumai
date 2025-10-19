//! Axum-specific server adapters
//!
//! This module provides utilities to convert `ChatStream` into Axum-compatible responses.
//!
//! ## Features
//!
//! - **SSE Response**: `to_sse_response()` converts `ChatStream` to `Sse<impl Stream>`
//! - **Text Response**: `to_text_response()` converts `ChatStream` to plain text stream
//! - **Error Handling**: Automatic error masking for production environments
//! - **Type Safety**: Strongly typed with Axum's SSE types
//!
//! ## Example
//!
//! ```rust,no_run
//! use axum::{Router, routing::get, response::sse::Sse};
//! use siumai::server_adapters::axum::to_sse_response;
//! use siumai::server_adapters::SseOptions;
//! use siumai::stream::ChatStream;
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
use futures::{Stream, StreamExt};

use crate::error::LlmError;
use crate::server_adapters::SseOptions;
use crate::stream::{ChatStream, ChatStreamEvent};

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
/// use siumai::server_adapters::axum::to_sse_response;
/// use siumai::server_adapters::SseOptions;
/// use siumai::stream::ChatStream;
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
/// use siumai::server_adapters::axum::to_text_stream;
/// use siumai::stream::ChatStream;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Usage;
    use crate::types::chat::{ChatResponse, MessageContent};
    use crate::types::common::ResponseMetadata;
    use futures::stream;

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

        // We can't directly collect Sse, so we test the underlying stream
        // by converting it back to a stream
        let _sse = to_sse_response(chat_stream, SseOptions::default());

        // For now, just verify it compiles and can be created
        // In a real test, you would send this to an HTTP response
    }

    #[tokio::test]
    async fn test_to_sse_response_error_masking() {
        let events = vec![Ok(ChatStreamEvent::Error {
            error: "Sensitive error message".to_string(),
        })];

        let chat_stream: ChatStream = Box::pin(stream::iter(events));
        let _sse = to_sse_response(chat_stream, SseOptions::production());

        // Verify it compiles with production options
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
        assert_eq!(collected.len(), 2); // Only 2 content deltas, usage is filtered out

        assert_eq!(collected[0].as_ref().unwrap(), "Hello");
        assert_eq!(collected[1].as_ref().unwrap(), " world");
    }
}
