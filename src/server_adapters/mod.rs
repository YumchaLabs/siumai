//! Server adapters: convert ChatStream into framework-specific responses
//!
//! This module provides utilities to convert `ChatStream` into responses
//! compatible with various web frameworks (Axum, Actix, Warp, etc.).
//!
//! ## Features
//!
//! - **Framework-agnostic helpers**: `text_stream()`, `sse_lines()`
//! - **Axum integration**: `axum::to_sse_response()` (requires `server-adapters` feature)
//! - **Error masking**: Configurable error message sanitization for production
//! - **Type-safe**: Strongly typed event streams with proper error handling
//!
//! ## Example (Framework-agnostic)
//!
//! ```rust,no_run
//! use siumai::server_adapters::{text_stream, sse_lines, SseOptions};
//! use siumai::stream::ChatStream;
//!
//! async fn handle_stream(stream: ChatStream) {
//!     // Convert to plain text stream
//!     let text = text_stream(stream);
//!
//!     // Or convert to SSE with options
//!     let sse = sse_lines(stream, SseOptions {
//!         include_usage: true,
//!         include_end: true,
//!         mask_errors: true,
//!     });
//! }
//! ```
//!
//! ## Example (Axum)
//!
//! ```rust,no_run
//! #[cfg(feature = "server-adapters")]
//! use siumai::server_adapters::axum::to_sse_response;
//! use siumai::stream::ChatStream;
//! use axum::response::sse::Sse;
//!
//! #[cfg(feature = "server-adapters")]
//! async fn chat_handler(stream: ChatStream) -> Sse<impl futures::Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>>> {
//!     to_sse_response(stream, Default::default())
//! }
//! ```

use std::pin::Pin;

use futures::Stream;
use futures::StreamExt;

use crate::error::LlmError;
use crate::stream::{ChatStream, ChatStreamEvent};

#[cfg(feature = "server-adapters")]
pub mod axum;

/// Convert a `ChatStream` into a plain text stream of deltas.
/// Each ContentDelta is yielded as-is; Usage/Start/End are ignored.
pub fn text_stream(
    mut stream: ChatStream,
) -> Pin<Box<dyn Stream<Item = Result<String, LlmError>> + Send>> {
    let s = async_stream::try_stream! {
        while let Some(item) = stream.next().await {
            match item? {
                ChatStreamEvent::ContentDelta { delta, .. } => { yield delta; }
                ChatStreamEvent::Error { error } => { Err(LlmError::InternalError(error))?; }
                _ => {}
            }
        }
    };
    Box::pin(s)
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

/// Convert a `ChatStream` into SSE lines ("event: X\n" + "data: ...\n\n").
///
/// The consumer can write each yielded string chunk to the HTTP response.
///
/// ## Events
///
/// - `event: start` - Stream metadata (if `include_start` is true)
/// - `event: delta` - Content deltas with `{"delta": string, "index": number}`
/// - `event: tool` - Tool call deltas with `{"id": string, "name": string, "arguments_delta": string, "index": number}`
/// - `event: reasoning` - Thinking/reasoning deltas with `{"delta": string}`
/// - `event: usage` - Token usage updates (if `include_usage` is true)
/// - `event: end` - Final response (if `include_end` is true)
/// - `event: error` - Error events (masked if `mask_errors` is true)
///
/// ## Example
///
/// ```rust,no_run
/// use siumai::server_adapters::{sse_lines, SseOptions};
/// use siumai::stream::ChatStream;
/// use futures::StreamExt;
///
/// async fn example(stream: ChatStream) {
///     let sse = sse_lines(stream, SseOptions::production());
///
///     futures::pin_mut!(sse);
///     while let Some(chunk) = sse.next().await {
///         match chunk {
///             Ok(line) => println!("{}", line),
///             Err(e) => eprintln!("Error: {}", e),
///         }
///     }
/// }
/// ```
pub fn sse_lines(
    stream: ChatStream,
    opts: SseOptions,
) -> Pin<Box<dyn Stream<Item = Result<String, LlmError>> + Send>> {
    let s = async_stream::try_stream! {
        futures::pin_mut!(stream);
        while let Some(item) = stream.next().await {
            match item? {
                ChatStreamEvent::StreamStart { metadata } => {
                    if opts.include_start {
                        let data = serde_json::to_string(&metadata).unwrap_or("{}".into());
                        yield format!("event: start\ndata: {}\n\n", data);
                    }
                }
                ChatStreamEvent::ContentDelta { delta, index } => {
                    let data = serde_json::json!({"delta": delta, "index": index});
                    yield format!("event: delta\ndata: {}\n\n", data.to_string());
                }
                ChatStreamEvent::ToolCallDelta { id, function_name, arguments_delta, index } => {
                    let data = serde_json::json!({
                        "id": id, "name": function_name, "arguments_delta": arguments_delta, "index": index
                    });
                    yield format!("event: tool\ndata: {}\n\n", data.to_string());
                }
                ChatStreamEvent::ThinkingDelta { delta } => {
                    let data = serde_json::json!({"delta": delta});
                    yield format!("event: reasoning\ndata: {}\n\n", data.to_string());
                }
                ChatStreamEvent::UsageUpdate { usage } => {
                    if opts.include_usage {
                        let data = serde_json::to_string(&usage).unwrap_or("{}".into());
                        yield format!("event: usage\ndata: {}\n\n", data);
                    }
                }
                ChatStreamEvent::StreamEnd { response } => {
                    if opts.include_end {
                        let data = serde_json::to_string(&response).unwrap_or("{}".into());
                        yield format!("event: end\ndata: {}\n\n", data);
                    }
                }
                ChatStreamEvent::Error { error } => {
                    let msg = if opts.mask_errors {
                        opts.masked_error_message
                            .clone()
                            .unwrap_or_else(|| "internal error".to_string())
                    } else {
                        error
                    };
                    yield format!("event: error\ndata: {}\n\n", serde_json::json!({"error": msg}));
                    // also return an error to allow upstream to stop if desired
                    Err::<(), LlmError>(LlmError::InternalError(msg))?;
                }
            }
        }
    };
    Box::pin(s)
}
