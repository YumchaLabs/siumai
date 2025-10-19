//! Server adapters: convert ChatStream into text/SSE friendly streams
//!
//! These helpers are framework-agnostic and return
//! generic `Stream<Item = Result<String, LlmError>>` which can be written to
//! Axum/Actix/Warp responses by the application.

use std::pin::Pin;

use futures::Stream;
use futures::StreamExt;

use crate::error::LlmError;
use crate::stream::{ChatStream, ChatStreamEvent};

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
pub struct SseOptions {
    /// Whether to include usage updates frames.
    pub include_usage: bool,
    /// Whether to include a final `end` event with the full response JSON.
    pub include_end: bool,
    /// Whether to mask error messages.
    pub mask_errors: bool,
}

impl Default for SseOptions {
    fn default() -> Self {
        Self {
            include_usage: true,
            include_end: true,
            mask_errors: true,
        }
    }
}

/// Convert a `ChatStream` into SSE lines ("event: X\n" + "data: ...\n\n").
/// The consumer can write each yielded string chunk to the HTTP response.
pub fn sse_lines(
    stream: ChatStream,
    opts: SseOptions,
) -> Pin<Box<dyn Stream<Item = Result<String, LlmError>> + Send>> {
    let s = async_stream::try_stream! {
        futures::pin_mut!(stream);
        while let Some(item) = stream.next().await {
            match item? {
                ChatStreamEvent::StreamStart { metadata } => {
                    let data = serde_json::to_string(&metadata).unwrap_or("{}".into());
                    yield format!("event: start\ndata: {}\n\n", data);
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
                    let msg = if opts.mask_errors { "internal error".to_string() } else { error };
                    yield format!("event: error\ndata: {}\n\n", serde_json::json!({"error": msg}));
                    // also return an error to allow upstream to stop if desired
                    Err::<(), LlmError>(LlmError::InternalError(msg))?;
                }
            }
        }
    };
    Box::pin(s)
}
