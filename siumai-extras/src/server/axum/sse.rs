//! Basic Axum adapters for unified `ChatStream`.
//!
//! English-only comments in code as requested.

use std::convert::Infallible;
use std::pin::Pin;

use axum::{
    body::Body,
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::{
        Response,
        sse::{Event, Sse},
    },
};
use futures::{Stream, StreamExt};
use serde::Serialize;

use siumai::prelude::unified::{ChatStream, ChatStreamEvent, LlmError};

#[derive(Serialize)]
struct SsePartEnvelope<'a> {
    part: &'a siumai::prelude::unified::ChatStreamPart,
    replay: Option<&'a siumai::prelude::unified::ChatStreamReplay>,
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

/// Options for plain text stream response encoding.
#[derive(Debug, Clone)]
pub struct TextStreamResponseOptions {
    /// HTTP status code for the response.
    ///
    /// Default: `200 OK`
    pub status: StatusCode,

    /// Additional response headers.
    ///
    /// If `content-type` is absent, `text/plain; charset=utf-8` is inserted.
    pub headers: HeaderMap,
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

impl Default for TextStreamResponseOptions {
    fn default() -> Self {
        Self {
            status: StatusCode::OK,
            headers: HeaderMap::new(),
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

fn encode_part_event(
    part: &siumai::prelude::unified::ChatStreamPart,
    replay: Option<&siumai::prelude::unified::ChatStreamReplay>,
) -> Event {
    let data = serde_json::to_string(&SsePartEnvelope { part, replay })
        .unwrap_or_else(|_| "{}".to_string());
    Event::default().event("part").data(data)
}

/// Convert a `ChatStream` into an Axum SSE response.
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
            Ok(ChatStreamEvent::Part { part }) => Some(encode_part_event(&part, None)),
            Ok(ChatStreamEvent::PartWithReplay { part, replay }) => {
                Some(encode_part_event(&part, Some(&replay)))
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
                // Forward custom events as-is.
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

        // Convert Option<Event> to Result<Event, Infallible>.
        // If None, we skip this event by returning an empty event.
        Ok(event.unwrap_or_else(|| Event::default().comment("skipped")))
    });

    Sse::new(event_stream)
}

/// Convert a `ChatStream` into a plain text stream for Axum.
pub fn to_text_stream(
    stream: ChatStream,
) -> Pin<Box<dyn Stream<Item = Result<String, Infallible>> + Send>> {
    let text_stream = stream.filter_map(|item| async move {
        match item {
            Ok(ChatStreamEvent::ContentDelta { delta, .. }) => Some(Ok(delta)),
            Ok(ChatStreamEvent::Part {
                part: siumai::prelude::unified::ChatStreamPart::TextDelta { delta, .. },
            })
            | Ok(ChatStreamEvent::PartWithReplay {
                part: siumai::prelude::unified::ChatStreamPart::TextDelta { delta, .. },
                ..
            }) => Some(Ok(delta)),
            Ok(ChatStreamEvent::Error { error }) => Some(Ok(format!("\n[Error: {}]\n", error))),
            Err(e) => Some(Ok(format!("\n[Error: {}]\n", e.user_message()))),
            _ => None,
        }
    });

    Box::pin(text_stream)
}

/// Convert a `ChatStream` into an Axum plain text streaming response.
///
/// This is the Axum-specific equivalent of AI SDK `createTextStreamResponse`: each text chunk is
/// emitted as UTF-8 bytes and the response defaults to `content-type: text/plain; charset=utf-8`.
pub fn to_text_stream_response(stream: ChatStream) -> Response<Body> {
    to_text_stream_response_with_options(stream, TextStreamResponseOptions::default())
}

/// Convert a `ChatStream` into an Axum plain text streaming response with custom response options.
pub fn to_text_stream_response_with_options(
    stream: ChatStream,
    mut opts: TextStreamResponseOptions,
) -> Response<Body> {
    if !opts.headers.contains_key(header::CONTENT_TYPE) {
        opts.headers.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/plain; charset=utf-8"),
        );
    }

    let mut response = Response::new(Body::from_stream(to_text_stream(stream)));
    *response.status_mut() = opts.status;
    *response.headers_mut() = opts.headers;
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::response::IntoResponse;
    use futures::stream;
    use siumai::prelude::unified::{
        ChatResponse, ChatStreamPart, ChatStreamReplay, MessageContent, ResponseMetadata, Usage,
    };

    async fn sse_body_text(stream: ChatStream, opts: SseOptions) -> String {
        let resp = to_sse_response(stream, opts).into_response();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("body bytes");
        String::from_utf8(body.to_vec()).expect("utf8")
    }

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
                    headers: None,
                    body: None,
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
            Ok(ChatStreamEvent::UsageUpdate {
                usage: Usage::new(1, 2),
            }),
            Ok(ChatStreamEvent::StreamEnd {
                response: ChatResponse::new(MessageContent::Text("Hello world".to_string())),
            }),
        ];

        let chat_stream: ChatStream = Box::pin(stream::iter(events));
        let text = sse_body_text(chat_stream, SseOptions::default()).await;
        assert!(text.contains("event: start"));
        assert!(text.contains("event: delta"));
        assert!(text.contains("event: usage"));
        assert!(text.contains("event: end"));
    }

    #[tokio::test]
    async fn test_to_text_stream_basic() {
        let events = vec![
            Ok(ChatStreamEvent::ContentDelta {
                delta: "Hello".to_string(),
                index: None,
            }),
            Ok(ChatStreamEvent::ContentDelta {
                delta: " world".to_string(),
                index: None,
            }),
        ];

        let chat_stream: ChatStream = Box::pin(stream::iter(events));
        let mut text_stream = to_text_stream(chat_stream);

        let mut out = String::new();
        while let Some(item) = text_stream.next().await {
            out.push_str(&item.unwrap());
        }
        assert_eq!(out, "Hello world");
    }

    #[tokio::test]
    async fn test_to_text_stream_response_sets_text_plain_header() {
        let events = vec![
            Ok(ChatStreamEvent::ContentDelta {
                delta: "Hello".to_string(),
                index: None,
            }),
            Ok(ChatStreamEvent::ContentDelta {
                delta: " world".to_string(),
                index: None,
            }),
        ];

        let chat_stream: ChatStream = Box::pin(stream::iter(events));
        let response = to_text_stream_response(chat_stream);
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE),
            Some(&HeaderValue::from_static("text/plain; charset=utf-8"))
        );

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body bytes");
        assert_eq!(
            String::from_utf8(body.to_vec()).expect("utf8"),
            "Hello world"
        );
    }

    #[tokio::test]
    async fn test_to_text_stream_response_with_options_preserves_custom_headers() {
        let events = vec![Ok(ChatStreamEvent::ContentDelta {
            delta: "Accepted".to_string(),
            index: None,
        })];
        let mut headers = HeaderMap::new();
        headers.insert("x-test", HeaderValue::from_static("ok"));

        let chat_stream: ChatStream = Box::pin(stream::iter(events));
        let response = to_text_stream_response_with_options(
            chat_stream,
            TextStreamResponseOptions {
                status: StatusCode::ACCEPTED,
                headers,
            },
        );

        assert_eq!(response.status(), StatusCode::ACCEPTED);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE),
            Some(&HeaderValue::from_static("text/plain; charset=utf-8"))
        );
        assert_eq!(
            response.headers().get("x-test"),
            Some(&HeaderValue::from_static("ok"))
        );
    }

    #[tokio::test]
    async fn test_to_text_stream_reads_stable_text_delta_parts() {
        let events = vec![
            Ok(ChatStreamEvent::Part {
                part: ChatStreamPart::TextDelta {
                    id: "txt_1".to_string(),
                    delta: "Hello".to_string(),
                    provider_metadata: None,
                },
            }),
            Ok(ChatStreamEvent::PartWithReplay {
                part: ChatStreamPart::TextDelta {
                    id: "txt_1".to_string(),
                    delta: " world".to_string(),
                    provider_metadata: None,
                },
                replay: ChatStreamReplay::openai_responses(Some(1), None).expect("replay"),
            }),
        ];

        let chat_stream: ChatStream = Box::pin(stream::iter(events));
        let mut text_stream = to_text_stream(chat_stream);

        let mut out = String::new();
        while let Some(item) = text_stream.next().await {
            out.push_str(&item.unwrap());
        }
        assert_eq!(out, "Hello world");
    }

    #[tokio::test]
    async fn test_to_sse_response_wraps_part_events_in_stable_envelope() {
        let events = vec![Ok(ChatStreamEvent::Part {
            part: ChatStreamPart::TextDelta {
                id: "txt_1".to_string(),
                delta: "hello".to_string(),
                provider_metadata: None,
            },
        })];

        let chat_stream: ChatStream = Box::pin(stream::iter(events));
        let text = sse_body_text(chat_stream, SseOptions::minimal()).await;

        assert!(text.contains("event: part"));
        assert!(text.contains(
            r#"data: {"part":{"type":"text-delta","id":"txt_1","delta":"hello"},"replay":null}"#
        ));
    }

    #[tokio::test]
    async fn test_to_sse_response_keeps_replay_inside_same_part_envelope() {
        let events = vec![Ok(ChatStreamEvent::PartWithReplay {
            part: ChatStreamPart::ToolCall(siumai::prelude::unified::ChatStreamToolCall {
                tool_call_id: "call_1".to_string(),
                tool_name: "web_search".to_string(),
                input: "{}".to_string(),
                provider_executed: Some(true),
                dynamic: Some(true),
                provider_metadata: None,
            }),
            replay: ChatStreamReplay::openai_responses(
                Some(3),
                Some(serde_json::json!({ "id": "fc_1" })),
            )
            .expect("replay"),
        })];

        let chat_stream: ChatStream = Box::pin(stream::iter(events));
        let text = sse_body_text(chat_stream, SseOptions::minimal()).await;

        assert!(text.contains("event: part"));
        assert!(text.contains(r#""part":{"type":"tool-call","toolCallId":"call_1","toolName":"web_search","input":"{}","providerExecuted":true,"dynamic":true}"#));
        assert!(
            text.contains(
                r#""replay":{"openaiResponses":{"outputIndex":3,"rawItem":{"id":"fc_1"}}}"#
            )
        );
    }
}
