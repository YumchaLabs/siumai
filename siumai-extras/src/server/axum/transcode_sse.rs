//! Provider-native SSE gateway helpers (Axum).
//!
//! English-only comments in code as requested.

use std::convert::Infallible;
use std::pin::Pin;

use axum::{body::Body, http::header, response::Response};
use futures::{Stream, StreamExt};

use siumai::prelude::unified::{ChatStream, ChatStreamEvent};

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

/// Convert a `ChatStream` into an OpenAI Responses-compatible SSE byte stream.
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
    use siumai::experimental::streaming::encode_chat_stream_as_sse;
    use siumai::protocol::gemini::streaming::GeminiEventConverter;
    use siumai::protocol::gemini::types::GeminiConfig;

    let converter = GeminiEventConverter::new(GeminiConfig::default());

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
                            serde_json::json!({ "error": { "message": e.user_message() } })
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
        use siumai::prelude::unified::{ChatResponse, MessageContent};

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
}
