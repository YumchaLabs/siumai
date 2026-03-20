//! Stream encoders for provider-native wire formats.
//!
//! This module complements the existing *parsing* pipeline (provider stream -> `ChatStreamEvent`)
//! with *encoding* helpers (`ChatStreamEvent` -> provider stream bytes).

use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use std::pin::Pin;
use std::sync::Arc;

use crate::error::LlmError;
use crate::streaming::{ChatStreamEvent, JsonEventConverter, SseEventConverter, StreamProcessor};
use crate::types::{FinishReason, ResponseMetadata};

/// Byte stream suitable for HTTP responses (SSE/JSONL).
pub type ChatByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, LlmError>> + Send>>;

/// Transform a unified chat event stream by expanding (or dropping) events.
///
/// This is useful for gateway/proxy use-cases where provider-specific `Custom`
/// events need to be bridged into another provider's expected shape before
/// re-serialization.
pub fn transform_chat_event_stream<S, F>(
    stream: S,
    mut transform: F,
) -> Pin<Box<dyn Stream<Item = Result<ChatStreamEvent, LlmError>> + Send>>
where
    S: Stream<Item = Result<ChatStreamEvent, LlmError>> + Send + 'static,
    F: FnMut(ChatStreamEvent) -> Vec<ChatStreamEvent> + Send + 'static,
{
    Box::pin(stream.flat_map(move |item| {
        let out: Vec<Result<ChatStreamEvent, LlmError>> = match item {
            Ok(ev) => transform(ev).into_iter().map(Ok).collect(),
            Err(e) => vec![Err(e)],
        };
        futures_util::stream::iter(out)
    }))
}

/// Ensure a unified stream terminates with a `StreamEnd` event on clean EOF.
///
/// This is intended for bridge/gateway serialization paths where the upstream
/// event stream may end without an explicit terminal event. If the stream ends
/// cleanly without `StreamEnd`, `Error`, or transport errors, a synthetic
/// `StreamEnd` with `finish_reason = Unknown` is appended.
pub fn ensure_stream_end<S>(
    stream: S,
) -> Pin<Box<dyn Stream<Item = Result<ChatStreamEvent, LlmError>> + Send>>
where
    S: Stream<Item = Result<ChatStreamEvent, LlmError>> + Send + 'static,
{
    Box::pin(async_stream::stream! {
        let mut upstream = Box::pin(stream);
        let mut processor = StreamProcessor::new();
        let mut saw_stream_end = false;
        let mut saw_error_event = false;
        let mut saw_transport_error = false;
        let mut stream_start_metadata: Option<ResponseMetadata> = None;

        while let Some(item) = upstream.next().await {
            match item {
                Ok(event) => {
                    if let ChatStreamEvent::StreamStart { metadata } = &event {
                        stream_start_metadata = Some(metadata.clone());
                    }
                    if matches!(event, ChatStreamEvent::StreamEnd { .. }) {
                        saw_stream_end = true;
                    }
                    if matches!(event, ChatStreamEvent::Error { .. }) {
                        saw_error_event = true;
                    }
                    let _ = processor.process_event(event.clone());
                    yield Ok(event);
                }
                Err(error) => {
                    saw_transport_error = true;
                    yield Err(error);
                }
            }
        }

        if !saw_stream_end && !saw_error_event && !saw_transport_error {
            let mut response =
                processor.build_final_response_with_finish_reason(Some(FinishReason::Unknown));
            apply_stream_start_metadata(&mut response, stream_start_metadata);
            yield Ok(ChatStreamEvent::StreamEnd { response });
        }
    })
}

fn apply_stream_start_metadata(
    response: &mut crate::types::ChatResponse,
    metadata: Option<ResponseMetadata>,
) {
    let Some(metadata) = metadata else {
        return;
    };

    if response.id.is_none() {
        response.id = metadata.id;
    }
    if response.model.is_none() {
        response.model = metadata.model;
    }
}

/// Encode a unified chat event stream into provider-native SSE frames.
///
/// Each `ChatStreamEvent` is encoded via `SseEventConverter::serialize_event`.
/// Empty byte vectors are treated as "no output" and are skipped.
pub fn encode_chat_stream_as_sse<C, S>(stream: S, converter: C) -> ChatByteStream
where
    C: SseEventConverter + Send + Sync + 'static,
    S: Stream<Item = Result<ChatStreamEvent, LlmError>> + Send + 'static,
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
    S: Stream<Item = Result<ChatStreamEvent, LlmError>> + Send + 'static,
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

#[cfg(test)]
mod tests {
    use super::*;

    use futures_util::stream;

    use crate::types::{MessageContent, Usage};

    #[tokio::test]
    async fn ensure_stream_end_appends_unknown_finish_reason_on_clean_eof() {
        let stream = stream::iter(vec![
            Ok(ChatStreamEvent::StreamStart {
                metadata: ResponseMetadata {
                    id: Some("resp_1".to_string()),
                    model: Some("test-model".to_string()),
                    created: None,
                    provider: "test".to_string(),
                    request_id: None,
                },
            }),
            Ok(ChatStreamEvent::ContentDelta {
                delta: "Hello".to_string(),
                index: None,
            }),
            Ok(ChatStreamEvent::UsageUpdate {
                usage: Usage::new(1, 2),
            }),
        ]);

        let events: Vec<_> = ensure_stream_end(stream).collect().await;
        assert_eq!(events.len(), 4);
        match events
            .last()
            .expect("last event")
            .as_ref()
            .expect("ok event")
        {
            ChatStreamEvent::StreamEnd { response } => {
                assert_eq!(response.id.as_deref(), Some("resp_1"));
                assert_eq!(response.model.as_deref(), Some("test-model"));
                assert_eq!(response.finish_reason, Some(FinishReason::Unknown));
                assert_eq!(response.usage.as_ref().map(|u| u.total_tokens), Some(3));
                assert_eq!(response.content, MessageContent::Text("Hello".to_string()));
            }
            other => panic!("expected synthetic StreamEnd, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn ensure_stream_end_does_not_duplicate_existing_terminal_event() {
        let stream = stream::iter(vec![Ok(ChatStreamEvent::StreamEnd {
            response: crate::types::ChatResponse::empty_with_finish_reason(FinishReason::Stop),
        })]);

        let events: Vec<_> = ensure_stream_end(stream).collect().await;
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0].as_ref().expect("ok event"),
            ChatStreamEvent::StreamEnd { response }
                if response.finish_reason == Some(FinishReason::Stop)
        ));
    }

    #[tokio::test]
    async fn ensure_stream_end_does_not_finalize_error_streams() {
        let stream = stream::iter(vec![Ok(ChatStreamEvent::Error {
            error: "boom".to_string(),
        })]);

        let events: Vec<_> = ensure_stream_end(stream).collect().await;
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0].as_ref().expect("ok event"),
            ChatStreamEvent::Error { .. }
        ));
    }
}
