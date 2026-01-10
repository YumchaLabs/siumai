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

/// Options for transcoding a `ChatStream` into a provider SSE wire format.
#[derive(Debug, Clone)]
pub struct TranscodeSseOptions {
    /// Controls lossy fallback for v3 parts that do not have a native representation
    /// in the target protocol stream.
    pub v3_unsupported_part_behavior: siumai::experimental::streaming::V3UnsupportedPartBehavior,
    /// Whether to bridge multiplexed Vercel-aligned tool parts into richer OpenAI Responses
    /// output_item frames (adds rawItem scaffolding when possible).
    pub bridge_openai_responses_stream_parts: bool,
}

impl Default for TranscodeSseOptions {
    fn default() -> Self {
        Self {
            v3_unsupported_part_behavior:
                siumai::experimental::streaming::V3UnsupportedPartBehavior::Drop,
            bridge_openai_responses_stream_parts: true,
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
    use siumai::experimental::streaming::{
        OpenAiResponsesStreamPartsBridge, encode_chat_stream_as_sse, transform_chat_event_stream,
    };
    use siumai::protocol::openai::responses_sse::OpenAiResponsesEventConverter;

    let mut bridge = OpenAiResponsesStreamPartsBridge::new();

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
    let body = Body::from_stream(to_openai_responses_sse_stream(stream));
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

    let converter = OpenAiCompatibleEventConverter::new(cfg, adapter);

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
                    return to_openai_responses_sse_response(stream);
                }
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
                if opts.v3_unsupported_part_behavior
                    == siumai::experimental::streaming::V3UnsupportedPartBehavior::AsText
                {
                    use siumai::experimental::streaming::{
                        LanguageModelV3StreamPart, transform_chat_event_stream,
                    };

                    let stream = transform_chat_event_stream(stream, move |ev| match ev {
                        ChatStreamEvent::Custom { event_type, data } => {
                            let Some(part) = LanguageModelV3StreamPart::parse_loose_json(&data)
                            else {
                                return vec![ChatStreamEvent::Custom { event_type, data }];
                            };

                            let mut out = part.to_best_effort_chat_events();
                            if out.is_empty()
                                && let Some(text) = part.to_lossy_text()
                            {
                                out.push(ChatStreamEvent::ContentDelta {
                                    delta: text,
                                    index: None,
                                });
                            }

                            if out.is_empty() {
                                vec![ChatStreamEvent::Custom { event_type, data }]
                            } else {
                                out
                            }
                        }
                        other => vec![other],
                    });

                    return to_openai_chat_completions_sse_response(stream);
                }

                to_openai_chat_completions_sse_response(stream)
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
                    .with_v3_unsupported_part_behavior(opts.v3_unsupported_part_behavior);
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
