//! Provider-native SSE gateway helpers (Axum).
//!
//! English-only comments in code as requested.

use std::convert::Infallible;
use std::fmt;
use std::pin::Pin;
use std::time::Duration;

use axum::{
    body::{Body, Bytes},
    http::header,
    response::Response,
};
use futures::{Stream, StreamExt};
use tokio::time::Sleep;

use siumai::experimental::bridge::{
    BridgeMode, BridgeOptions, BridgeTarget, transform_chat_stream_with_bridge_options,
};
use siumai::prelude::unified::{ChatStream, ChatStreamEvent};

use crate::server::GatewayBridgePolicy;

type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, Infallible>> + Send>>;

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
#[derive(Clone)]
pub struct TranscodeSseOptions {
    /// Controls lossy fallback for v3 parts that do not have a native representation
    /// in the target protocol stream.
    pub v3_unsupported_part_behavior: siumai::experimental::streaming::V3UnsupportedPartBehavior,
    /// Whether to bridge multiplexed Vercel-aligned tool parts into richer OpenAI Responses
    /// output_item frames (adds rawItem scaffolding when possible).
    pub bridge_openai_responses_stream_parts: bool,
    /// Whether to serialize v3 tool results as Gemini `functionResponse` frames (gateway-only).
    pub gemini_emit_function_response_tool_results: bool,
    /// Optional bridge customization applied before target SSE serialization.
    pub bridge_options: Option<BridgeOptions>,
    /// Optional gateway bridge policy applied by the helper.
    pub policy: Option<GatewayBridgePolicy>,
}

impl fmt::Debug for TranscodeSseOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TranscodeSseOptions")
            .field(
                "v3_unsupported_part_behavior",
                &self.v3_unsupported_part_behavior,
            )
            .field(
                "bridge_openai_responses_stream_parts",
                &self.bridge_openai_responses_stream_parts,
            )
            .field(
                "gemini_emit_function_response_tool_results",
                &self.gemini_emit_function_response_tool_results,
            )
            .field("has_bridge_options", &self.bridge_options.is_some())
            .field("has_policy", &self.policy.is_some())
            .finish()
    }
}

impl Default for TranscodeSseOptions {
    fn default() -> Self {
        Self {
            v3_unsupported_part_behavior:
                siumai::experimental::streaming::V3UnsupportedPartBehavior::Drop,
            bridge_openai_responses_stream_parts: true,
            gemini_emit_function_response_tool_results: false,
            bridge_options: None,
            policy: None,
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

    /// Attach bridge customization options to the SSE transcode helper.
    pub fn with_bridge_options(mut self, bridge_options: BridgeOptions) -> Self {
        self.bridge_options = Some(bridge_options);
        self
    }

    /// Attach a gateway bridge policy to the SSE transcode helper.
    pub fn with_policy(mut self, policy: GatewayBridgePolicy) -> Self {
        self.policy = Some(policy);
        self
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
) -> ByteStream {
    build_target_sse_stream(stream, TargetSseFormat::OpenAiResponses, &opts)
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
    build_sse_response(
        to_openai_responses_sse_stream_with_options(stream, opts.clone()),
        TargetSseFormat::OpenAiResponses,
        &opts,
    )
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
    build_sse_response(
        build_target_sse_stream(stream, TargetSseFormat::OpenAiChatCompletions, &opts),
        TargetSseFormat::OpenAiChatCompletions,
        &opts,
    )
}

/// Convert a `ChatStream` into an Anthropic Messages SSE response (best-effort).
#[cfg(feature = "anthropic")]
pub fn to_anthropic_messages_sse_response(stream: ChatStream) -> Response<Body> {
    let opts = TranscodeSseOptions::default();
    build_sse_response(
        build_target_sse_stream(stream, TargetSseFormat::AnthropicMessages, &opts),
        TargetSseFormat::AnthropicMessages,
        &opts,
    )
}

/// Convert a `ChatStream` into a Google Gemini GenerateContent SSE response (best-effort).
#[cfg(feature = "google")]
pub fn to_gemini_generate_content_sse_response(stream: ChatStream) -> Response<Body> {
    let opts = TranscodeSseOptions::default();
    build_sse_response(
        build_target_sse_stream(stream, TargetSseFormat::GeminiGenerateContent, &opts),
        TargetSseFormat::GeminiGenerateContent,
        &opts,
    )
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
                to_openai_responses_sse_response_with_options(stream, opts)
            }
            #[cfg(not(feature = "openai"))]
            {
                let _ = opts;
                let _ = stream;
                feature_disabled_response("openai feature is disabled")
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
                feature_disabled_response("openai feature is disabled")
            }
        }
        TargetSseFormat::AnthropicMessages => {
            #[cfg(feature = "anthropic")]
            {
                build_sse_response(
                    build_target_sse_stream(stream, TargetSseFormat::AnthropicMessages, &opts),
                    TargetSseFormat::AnthropicMessages,
                    &opts,
                )
            }
            #[cfg(not(feature = "anthropic"))]
            {
                let _ = opts;
                let _ = stream;
                feature_disabled_response("anthropic feature is disabled")
            }
        }
        TargetSseFormat::GeminiGenerateContent => {
            #[cfg(feature = "google")]
            {
                build_sse_response(
                    build_target_sse_stream(stream, TargetSseFormat::GeminiGenerateContent, &opts),
                    TargetSseFormat::GeminiGenerateContent,
                    &opts,
                )
            }
            #[cfg(not(feature = "google"))]
            {
                let _ = opts;
                let _ = stream;
                feature_disabled_response("google feature is disabled")
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

fn apply_bridge_stream_options(
    stream: ChatStream,
    target: TargetSseFormat,
    opts: &TranscodeSseOptions,
) -> ChatStream {
    let bridge_options = opts
        .policy
        .as_ref()
        .map(|policy| policy.resolve_bridge_options(opts.bridge_options.as_ref()))
        .or_else(|| opts.bridge_options.clone());
    let Some(bridge_options) = bridge_options.as_ref() else {
        return stream;
    };

    transform_chat_stream_with_bridge_options(
        stream,
        None,
        bridge_target_for_sse(target),
        bridge_options,
        Some(match target {
            TargetSseFormat::OpenAiResponses => "axum-openai-responses-sse".to_string(),
            TargetSseFormat::OpenAiChatCompletions => {
                "axum-openai-chat-completions-sse".to_string()
            }
            TargetSseFormat::AnthropicMessages => "axum-anthropic-messages-sse".to_string(),
            TargetSseFormat::GeminiGenerateContent => {
                "axum-gemini-generate-content-sse".to_string()
            }
        }),
    )
}

#[cfg(any(feature = "openai", feature = "anthropic", feature = "google"))]
fn build_target_sse_stream(
    stream: ChatStream,
    target: TargetSseFormat,
    opts: &TranscodeSseOptions,
) -> ByteStream {
    let stream = apply_bridge_stream_options(stream, target, opts);
    let stream = match target {
        TargetSseFormat::OpenAiResponses => {
            #[cfg(feature = "openai")]
            {
                encode_openai_responses_sse_stream(stream, opts)
            }
            #[cfg(not(feature = "openai"))]
            unreachable!("openai target requires openai feature")
        }
        TargetSseFormat::OpenAiChatCompletions => {
            #[cfg(feature = "openai")]
            {
                encode_openai_chat_completions_sse_stream(stream, opts)
            }
            #[cfg(not(feature = "openai"))]
            unreachable!("openai target requires openai feature")
        }
        TargetSseFormat::AnthropicMessages => {
            #[cfg(feature = "anthropic")]
            {
                encode_anthropic_messages_sse_stream(stream, opts)
            }
            #[cfg(not(feature = "anthropic"))]
            unreachable!("anthropic target requires anthropic feature")
        }
        TargetSseFormat::GeminiGenerateContent => {
            #[cfg(feature = "google")]
            {
                encode_gemini_generate_content_sse_stream(stream, opts)
            }
            #[cfg(not(feature = "google"))]
            unreachable!("gemini target requires google feature")
        }
    };

    apply_runtime_stream_policy(stream, target, opts)
}

#[cfg(feature = "openai")]
fn encode_openai_responses_sse_stream(
    stream: ChatStream,
    opts: &TranscodeSseOptions,
) -> ByteStream {
    use siumai::experimental::streaming::{
        OpenAiResponsesStreamPartsBridge, encode_chat_stream_as_sse, transform_chat_event_stream,
    };
    use siumai::protocol::openai::responses_sse::OpenAiResponsesEventConverter;

    let stream = apply_openai_responses_v3_policy(stream, opts.v3_unsupported_part_behavior);
    let stream = if opts.bridge_openai_responses_stream_parts {
        let mut bridge = OpenAiResponsesStreamPartsBridge::new();
        transform_chat_event_stream(stream, move |ev| bridge.bridge_event(ev))
    } else {
        stream
    };

    let converter = OpenAiResponsesEventConverter::new();
    let bytes_stream = encode_chat_stream_as_sse(stream, converter);

    Box::pin(bytes_stream.map(|item| match item {
        Ok(bytes) => Ok(bytes),
        Err(e) => Ok(Bytes::from(format!(
            "event: response.error\ndata: {{\"type\":\"response.error\",\"error\":{{\"message\":{}}}}}\n\n",
            serde_json::json!(e.user_message())
        ))),
    }))
}

#[cfg(feature = "openai")]
fn encode_openai_chat_completions_sse_stream(
    stream: ChatStream,
    opts: &TranscodeSseOptions,
) -> ByteStream {
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

    Box::pin(bytes.map(|item| match item {
        Ok(bytes) => Ok(bytes),
        Err(e) => Ok(Bytes::from(format!(
            "event: error\ndata: {}\n\n",
            serde_json::json!({ "error": e.user_message() })
        ))),
    }))
}

#[cfg(feature = "anthropic")]
fn encode_anthropic_messages_sse_stream(
    stream: ChatStream,
    opts: &TranscodeSseOptions,
) -> ByteStream {
    use siumai::experimental::streaming::encode_chat_stream_as_sse;
    use siumai::protocol::anthropic::params::AnthropicParams;
    use siumai::protocol::anthropic::streaming::AnthropicEventConverter;

    let converter = AnthropicEventConverter::new(AnthropicParams::default())
        .with_v3_unsupported_part_behavior(opts.v3_unsupported_part_behavior);
    let bytes = encode_chat_stream_as_sse(stream, converter);

    Box::pin(bytes.map(|item| match item {
        Ok(bytes) => Ok(bytes),
        Err(e) => Ok(Bytes::from(format!(
            "event: error\ndata: {}\n\n",
            serde_json::json!({
                "type": "error",
                "error": { "type": "api_error", "message": e.user_message() }
            })
        ))),
    }))
}

#[cfg(feature = "google")]
fn encode_gemini_generate_content_sse_stream(
    stream: ChatStream,
    opts: &TranscodeSseOptions,
) -> ByteStream {
    use siumai::experimental::streaming::encode_chat_stream_as_sse;
    use siumai::protocol::gemini::streaming::GeminiEventConverter;
    use siumai::protocol::gemini::types::GeminiConfig;

    let converter = GeminiEventConverter::new(GeminiConfig::default())
        .with_v3_unsupported_part_behavior(opts.v3_unsupported_part_behavior)
        .with_emit_function_response_tool_results(opts.gemini_emit_function_response_tool_results);
    let bytes = encode_chat_stream_as_sse(stream, converter);

    Box::pin(bytes.map(|item| match item {
        Ok(bytes) => Ok(bytes),
        Err(e) => Ok(Bytes::from(format!(
            "data: {}\n\n",
            serde_json::json!({ "error": { "message": e.user_message() } })
        ))),
    }))
}

fn build_sse_response(
    stream: ByteStream,
    target: TargetSseFormat,
    opts: &TranscodeSseOptions,
) -> Response<Body> {
    let mut response = Response::new(Body::from_stream(stream));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("text/event-stream"),
    );

    if let Some(policy) = opts.policy.as_ref() {
        apply_gateway_policy_headers(
            response.headers_mut(),
            policy,
            target,
            effective_bridge_mode(opts).unwrap_or(policy.bridge_options.mode),
        );
    }

    response
}

#[allow(dead_code)]
fn feature_disabled_response(message: &'static str) -> Response<Body> {
    Response::builder()
        .status(501)
        .body(Body::from(message))
        .unwrap()
}

fn effective_bridge_mode(opts: &TranscodeSseOptions) -> Option<BridgeMode> {
    opts.policy
        .as_ref()
        .map(|policy| {
            policy
                .resolve_bridge_options(opts.bridge_options.as_ref())
                .mode
        })
        .or_else(|| opts.bridge_options.as_ref().map(|options| options.mode))
}

fn apply_runtime_stream_policy(
    stream: ByteStream,
    target: TargetSseFormat,
    opts: &TranscodeSseOptions,
) -> ByteStream {
    let Some(policy) = opts.policy.clone() else {
        return stream;
    };

    let keepalive_interval = policy.keepalive_interval;
    let idle_timeout = policy.stream_idle_timeout;
    if keepalive_interval.is_none() && idle_timeout.is_none() {
        return stream;
    }

    Box::pin(async_stream::stream! {
        let mut upstream = stream.fuse();
        let mut keepalive_sleep = keepalive_interval.map(sleep_after);
        let mut idle_sleep = idle_timeout.map(sleep_after);

        loop {
            let keepalive_tick = async {
                match keepalive_sleep.as_mut() {
                    Some(timer) => {
                        timer.await;
                    }
                    None => std::future::pending::<()>().await,
                }
            };
            let idle_tick = async {
                match idle_sleep.as_mut() {
                    Some(timer) => {
                        timer.await;
                    }
                    None => std::future::pending::<()>().await,
                }
            };

            tokio::select! {
                biased;
                item = upstream.next() => match item {
                    Some(item) => {
                        if let Some(interval) = keepalive_interval {
                            keepalive_sleep = Some(sleep_after(interval));
                        }
                        if let Some(timeout) = idle_timeout {
                            idle_sleep = Some(sleep_after(timeout));
                        }
                        yield item;
                    }
                    None => break,
                },
                _ = idle_tick => {
                    yield Ok(timeout_error_frame(
                        target,
                        idle_timeout_error_message(idle_timeout.expect("idle timeout configured"), policy.passthrough_runtime_errors),
                    ));
                    break;
                }
                _ = keepalive_tick => {
                    if let Some(interval) = keepalive_interval {
                        keepalive_sleep = Some(sleep_after(interval));
                        yield Ok(Bytes::from_static(b": keep-alive\n\n"));
                    }
                }
            }
        }
    })
}

fn sleep_after(duration: Duration) -> Pin<Box<Sleep>> {
    Box::pin(tokio::time::sleep(duration))
}

fn idle_timeout_error_message(timeout: Duration, passthrough: bool) -> String {
    if passthrough {
        format!("stream idle timeout after {} ms", timeout.as_millis())
    } else {
        "gateway stream idle timeout".to_string()
    }
}

fn timeout_error_frame(target: TargetSseFormat, message: String) -> Bytes {
    Bytes::from(match target {
        TargetSseFormat::OpenAiResponses => format!(
            "event: response.error\ndata: {{\"type\":\"response.error\",\"error\":{{\"message\":{}}}}}\n\n",
            serde_json::json!(message)
        ),
        TargetSseFormat::OpenAiChatCompletions => format!(
            "event: error\ndata: {}\n\n",
            serde_json::json!({ "error": { "message": message } })
        ),
        TargetSseFormat::AnthropicMessages => format!(
            "event: error\ndata: {}\n\n",
            serde_json::json!({
                "type": "error",
                "error": { "type": "api_error", "message": message }
            })
        ),
        TargetSseFormat::GeminiGenerateContent => format!(
            "data: {}\n\n",
            serde_json::json!({ "error": { "message": message } })
        ),
    })
}

fn apply_gateway_policy_headers(
    headers: &mut axum::http::HeaderMap,
    policy: &GatewayBridgePolicy,
    target: TargetSseFormat,
    mode: siumai::experimental::bridge::BridgeMode,
) {
    if !policy.emit_bridge_headers {
        return;
    }

    insert_policy_header(
        headers,
        policy,
        "x-siumai-bridge-target",
        match target {
            TargetSseFormat::OpenAiResponses => "openai-responses",
            TargetSseFormat::OpenAiChatCompletions => "openai-chat-completions",
            TargetSseFormat::AnthropicMessages => "anthropic-messages",
            TargetSseFormat::GeminiGenerateContent => "gemini-generate-content",
        },
    );
    insert_policy_header(
        headers,
        policy,
        "x-siumai-bridge-mode",
        match mode {
            siumai::experimental::bridge::BridgeMode::Strict => "strict",
            siumai::experimental::bridge::BridgeMode::BestEffort => "best-effort",
            siumai::experimental::bridge::BridgeMode::ProviderTolerant => "provider-tolerant",
        },
    );
}

fn insert_policy_header(
    headers: &mut axum::http::HeaderMap,
    policy: &GatewayBridgePolicy,
    name: &'static str,
    value: &str,
) {
    if !policy.allows_response_header(name) {
        return;
    }
    if let Ok(value) = axum::http::HeaderValue::from_str(value) {
        headers.insert(name, value);
    }
}

fn bridge_target_for_sse(target: TargetSseFormat) -> BridgeTarget {
    match target {
        TargetSseFormat::OpenAiResponses => BridgeTarget::OpenAiResponses,
        TargetSseFormat::OpenAiChatCompletions => BridgeTarget::OpenAiChatCompletions,
        TargetSseFormat::AnthropicMessages => BridgeTarget::AnthropicMessages,
        TargetSseFormat::GeminiGenerateContent => BridgeTarget::GeminiGenerateContent,
    }
}

#[cfg(test)]
mod transcode_tests {
    use super::*;

    use std::time::Duration;

    use siumai::experimental::bridge::{BridgeMode, BridgeOptions};

    use super::super::bridge_hooks::stream_bridge_hook;

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

    #[tokio::test]
    async fn transcode_sse_bridge_options_can_transform_events() {
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

        let resp = to_transcoded_sse_response(
            chat_stream,
            TargetSseFormat::OpenAiResponses,
            TranscodeSseOptions::default().with_bridge_options(
                BridgeOptions::new(BridgeMode::BestEffort)
                    .with_route_label("tests.axum.sse.transform")
                    .with_stream_hook(stream_bridge_hook(|_, event| match event {
                        ChatStreamEvent::ContentDelta { delta, index } => {
                            vec![ChatStreamEvent::ContentDelta {
                                delta: delta.to_uppercase(),
                                index,
                            }]
                        }
                        other => vec![other],
                    })),
            ),
        );

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("body bytes");
        let text = String::from_utf8(body.to_vec()).expect("utf8");
        assert!(text.contains("HELLO"));
    }

    #[tokio::test]
    async fn transcode_sse_policy_emits_bridge_headers() {
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

        let resp = to_transcoded_sse_response(
            chat_stream,
            TargetSseFormat::OpenAiResponses,
            TranscodeSseOptions::default().with_policy(
                GatewayBridgePolicy::new(BridgeMode::BestEffort).with_bridge_headers(true),
            ),
        );

        assert_eq!(resp.headers()["x-siumai-bridge-target"], "openai-responses");
        assert_eq!(resp.headers()["x-siumai-bridge-mode"], "best-effort");
    }

    #[tokio::test]
    #[cfg(feature = "openai")]
    async fn transcode_sse_policy_emits_keepalive_comments() {
        use async_stream::stream;
        use siumai::prelude::unified::{ChatResponse, MessageContent};

        let chat_stream: ChatStream = Box::pin(stream! {
            tokio::time::sleep(Duration::from_millis(25)).await;
            yield Ok(ChatStreamEvent::ContentDelta {
                delta: "hello".to_string(),
                index: None,
            });
            yield Ok(ChatStreamEvent::StreamEnd {
                response: ChatResponse::new(MessageContent::Text("done".to_string())),
            });
        });

        let resp = to_transcoded_sse_response(
            chat_stream,
            TargetSseFormat::OpenAiResponses,
            TranscodeSseOptions::default().with_policy(
                GatewayBridgePolicy::new(BridgeMode::BestEffort)
                    .with_keepalive_interval(Duration::from_millis(5)),
            ),
        );

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("body bytes");
        let text = String::from_utf8(body.to_vec()).expect("utf8");
        assert!(text.contains(": keep-alive"));
        assert!(text.contains("hello"));
    }

    #[tokio::test]
    #[cfg(feature = "openai")]
    async fn transcode_sse_policy_times_out_idle_streams() {
        use async_stream::stream;

        let chat_stream: ChatStream = Box::pin(stream! {
            tokio::time::sleep(Duration::from_millis(30)).await;
            yield Ok(ChatStreamEvent::ContentDelta {
                delta: "late".to_string(),
                index: None,
            });
        });

        let resp = to_transcoded_sse_response(
            chat_stream,
            TargetSseFormat::OpenAiResponses,
            TranscodeSseOptions::default().with_policy(
                GatewayBridgePolicy::new(BridgeMode::BestEffort)
                    .with_stream_idle_timeout(Duration::from_millis(5)),
            ),
        );

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("body bytes");
        let text = String::from_utf8(body.to_vec()).expect("utf8");
        assert!(text.contains("response.error"));
        assert!(text.contains("idle timeout"));
        assert!(!text.contains("late"));
    }

    #[tokio::test]
    #[cfg(feature = "openai")]
    async fn openai_responses_direct_helper_applies_bridge_options() {
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

        let resp = to_openai_responses_sse_response_with_options(
            chat_stream,
            TranscodeSseOptions::default().with_bridge_options(
                BridgeOptions::new(BridgeMode::BestEffort)
                    .with_route_label("tests.axum.sse.direct-transform")
                    .with_stream_hook(stream_bridge_hook(|_, event| match event {
                        ChatStreamEvent::ContentDelta { delta, index } => {
                            vec![ChatStreamEvent::ContentDelta {
                                delta: delta.to_uppercase(),
                                index,
                            }]
                        }
                        other => vec![other],
                    })),
            ),
        );

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("body bytes");
        let text = String::from_utf8(body.to_vec()).expect("utf8");
        assert!(text.contains("HELLO"));
    }
}
