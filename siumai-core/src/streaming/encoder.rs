//! Stream encoders for provider-native wire formats.
//!
//! This module complements the existing *parsing* pipeline (provider stream -> `ChatStreamEvent`)
//! with *encoding* helpers (`ChatStreamEvent` -> provider stream bytes).

use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use std::pin::Pin;
use std::sync::Arc;

use crate::error::LlmError;
use crate::streaming::{ChatStreamEvent, JsonEventConverter, SseEventConverter};

/// Byte stream suitable for HTTP responses (SSE/JSONL).
pub type ChatByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, LlmError>> + Send + Sync>>;

/// Encode a unified chat event stream into provider-native SSE frames.
///
/// Each `ChatStreamEvent` is encoded via `SseEventConverter::serialize_event`.
/// Empty byte vectors are treated as "no output" and are skipped.
pub fn encode_chat_stream_as_sse<C, S>(stream: S, converter: C) -> ChatByteStream
where
    C: SseEventConverter + Send + Sync + 'static,
    S: Stream<Item = Result<ChatStreamEvent, LlmError>> + Send + Sync + 'static,
{
    let converter: Arc<dyn SseEventConverter> = Arc::new(converter);
    Box::pin(stream.filter_map(move |item| {
        let converter = converter.clone();
        async move {
            match item {
                Ok(ev) => match converter.serialize_event(&ev) {
                    Ok(bytes) if bytes.is_empty() => None,
                    Ok(bytes) => Some(Ok(Bytes::from(bytes))),
                    Err(e) => Some(Err(e)),
                },
                Err(e) => Some(Err(e)),
            }
        }
    }))
}

/// Encode a unified chat event stream into provider-native JSON Lines.
///
/// Each `ChatStreamEvent` is encoded via `JsonEventConverter::serialize_event`.
/// Empty byte vectors are treated as "no output" and are skipped.
pub fn encode_chat_stream_as_jsonl<C, S>(stream: S, converter: C) -> ChatByteStream
where
    C: JsonEventConverter + Send + Sync + 'static,
    S: Stream<Item = Result<ChatStreamEvent, LlmError>> + Send + Sync + 'static,
{
    let converter: Arc<dyn JsonEventConverter> = Arc::new(converter);
    Box::pin(stream.filter_map(move |item| {
        let converter = converter.clone();
        async move {
            match item {
                Ok(ev) => match converter.serialize_event(&ev) {
                    Ok(bytes) if bytes.is_empty() => None,
                    Ok(bytes) => Some(Ok(Bytes::from(bytes))),
                    Err(e) => Some(Err(e)),
                },
                Err(e) => Some(Err(e)),
            }
        }
    }))
}
