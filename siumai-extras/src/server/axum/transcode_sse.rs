//! Provider-native SSE gateway helpers (Axum).
//!
//! English-only comments in code as requested.

use std::convert::Infallible;
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use axum::{
    body::{Body, Bytes},
    http::header,
    response::Response,
};
use futures::{Stream, StreamExt};
use tokio::time::Sleep;

use siumai::experimental::bridge::{
    BridgeCustomization, BridgeLossAction, BridgeMode, BridgeOptions, BridgeOptionsOverride,
    BridgeReport, BridgeTarget, StreamBridgeContext, inspect_chat_stream_bridge,
    transform_chat_stream_with_bridge_options,
};
use siumai::prelude::unified::{ChatStream, ChatStreamEvent};

use crate::server::{
    GatewayBridgePolicy, gateway_bridge_headers, gateway_sse_runtime_policy,
    resolve_gateway_bridge_options,
};

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
    /// Optional known protocol view of the upstream stream before target transcoding.
    pub bridge_source: Option<BridgeTarget>,
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
    /// Optional partial bridge override applied on top of route/policy defaults.
    pub bridge_options_override: Option<BridgeOptionsOverride>,
    /// Optional gateway bridge policy applied by the helper.
    pub policy: Option<GatewayBridgePolicy>,
}

impl fmt::Debug for TranscodeSseOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TranscodeSseOptions")
            .field("bridge_source", &self.bridge_source)
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
            .field(
                "has_bridge_options_override",
                &self.bridge_options_override.is_some(),
            )
            .field("has_policy", &self.policy.is_some())
            .finish()
    }
}

impl Default for TranscodeSseOptions {
    fn default() -> Self {
        Self {
            bridge_source: None,
            v3_unsupported_part_behavior:
                siumai::experimental::streaming::V3UnsupportedPartBehavior::Drop,
            bridge_openai_responses_stream_parts: true,
            gemini_emit_function_response_tool_results: false,
            bridge_options: None,
            bridge_options_override: None,
            policy: None,
        }
    }
}

impl TranscodeSseOptions {
    /// Declare the known protocol view of the upstream stream.
    pub fn with_bridge_source(mut self, source: BridgeTarget) -> Self {
        self.bridge_source = Some(source);
        self
    }

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

    /// Attach a unified bridge customization object to the SSE transcode helper.
    pub fn with_bridge_customization(
        mut self,
        customization: Arc<dyn BridgeCustomization>,
    ) -> Self {
        self.bridge_options = Some(
            self.bridge_options
                .take()
                .unwrap_or_else(|| BridgeOptions::new(BridgeMode::BestEffort))
                .with_customization(customization),
        );
        self
    }

    /// Attach a partial bridge override to the SSE transcode helper.
    pub fn with_bridge_options_override(
        mut self,
        bridge_options_override: BridgeOptionsOverride,
    ) -> Self {
        self.bridge_options_override = Some(bridge_options_override);
        self
    }

    /// Override only the effective bridge mode used by the SSE transcode helper.
    pub fn with_bridge_mode_override(mut self, mode: BridgeMode) -> Self {
        let override_options = self
            .bridge_options_override
            .take()
            .unwrap_or_default()
            .with_mode(mode);
        self.bridge_options_override = Some(override_options);
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
    transcode_sse_response(stream, TargetSseFormat::OpenAiResponses, opts)
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
    transcode_sse_response(stream, TargetSseFormat::OpenAiChatCompletions, opts)
}

/// Convert a `ChatStream` into an Anthropic Messages SSE response (best-effort).
#[cfg(feature = "anthropic")]
pub fn to_anthropic_messages_sse_response(stream: ChatStream) -> Response<Body> {
    to_anthropic_messages_sse_response_with_options(stream, TranscodeSseOptions::default())
}

/// Convert a `ChatStream` into an Anthropic Messages SSE response with bridge options.
#[cfg(feature = "anthropic")]
pub fn to_anthropic_messages_sse_response_with_options(
    stream: ChatStream,
    opts: TranscodeSseOptions,
) -> Response<Body> {
    transcode_sse_response(stream, TargetSseFormat::AnthropicMessages, opts)
}

/// Convert a `ChatStream` into a Google Gemini GenerateContent SSE response (best-effort).
#[cfg(feature = "google")]
pub fn to_gemini_generate_content_sse_response(stream: ChatStream) -> Response<Body> {
    to_gemini_generate_content_sse_response_with_options(stream, TranscodeSseOptions::default())
}

/// Convert a `ChatStream` into a Google Gemini GenerateContent SSE response with bridge options.
#[cfg(feature = "google")]
pub fn to_gemini_generate_content_sse_response_with_options(
    stream: ChatStream,
    opts: TranscodeSseOptions,
) -> Response<Body> {
    transcode_sse_response(stream, TargetSseFormat::GeminiGenerateContent, opts)
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
                transcode_sse_response(stream, TargetSseFormat::OpenAiResponses, opts)
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
                transcode_sse_response(stream, TargetSseFormat::OpenAiChatCompletions, opts)
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
                transcode_sse_response(stream, TargetSseFormat::AnthropicMessages, opts)
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
                transcode_sse_response(stream, TargetSseFormat::GeminiGenerateContent, opts)
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

fn transcode_sse_response(
    stream: ChatStream,
    target: TargetSseFormat,
    opts: TranscodeSseOptions,
) -> Response<Body> {
    let report = match inspect_stream_bridge_for_response(target, &opts) {
        Ok(report) => report,
        Err(report) => return rejected_sse_response(report, opts.policy.as_ref()),
    };

    let stream = build_target_sse_stream(stream, target, &opts);
    build_sse_response(stream, target, &opts, report.as_ref())
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
    let bridge_options = resolve_bridge_options(opts);
    let Some(bridge_options) = bridge_options.as_ref() else {
        return stream;
    };

    transform_chat_stream_with_bridge_options(
        stream,
        opts.bridge_source,
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

fn inspect_stream_bridge_for_response(
    target: TargetSseFormat,
    opts: &TranscodeSseOptions,
) -> Result<Option<BridgeReport>, BridgeReport> {
    let Some(bridge_options) = resolve_bridge_options(opts) else {
        return Ok(None);
    };

    let target = bridge_target_for_sse(target);
    let path_label = match target {
        BridgeTarget::OpenAiResponses => "axum-openai-responses-sse",
        BridgeTarget::OpenAiChatCompletions => "axum-openai-chat-completions-sse",
        BridgeTarget::AnthropicMessages => "axum-anthropic-messages-sse",
        BridgeTarget::GeminiGenerateContent => "axum-gemini-generate-content-sse",
    };
    let ctx = StreamBridgeContext::new(
        opts.bridge_source,
        target,
        bridge_options.mode,
        bridge_options.route_label.clone(),
        Some(path_label.to_string()),
    );
    let mut report = BridgeReport::with_source(opts.bridge_source, target, bridge_options.mode);
    inspect_chat_stream_bridge(opts.bridge_source, target, &mut report);

    if report.is_rejected() {
        return Err(report);
    }

    if matches!(
        bridge_options.loss_policy.stream_action(&ctx, &report),
        BridgeLossAction::Reject
    ) {
        report.reject(format!(
            "bridge policy rejected stream conversion to {}",
            ctx.target.as_str()
        ));
        return Err(report);
    }

    Ok(Some(report))
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
        OpenAiResponsesStreamPartsBridge, encode_chat_stream_as_sse, ensure_stream_end,
        transform_chat_event_stream,
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
    let bytes_stream = encode_chat_stream_as_sse(ensure_stream_end(stream), converter);

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
    use siumai::experimental::streaming::{encode_chat_stream_as_sse, ensure_stream_end};
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
    let bytes = encode_chat_stream_as_sse(ensure_stream_end(stream), converter);

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
    use siumai::experimental::streaming::{encode_chat_stream_as_sse, ensure_stream_end};
    use siumai::protocol::anthropic::params::AnthropicParams;
    use siumai::protocol::anthropic::streaming::AnthropicEventConverter;

    let converter = AnthropicEventConverter::new(AnthropicParams::default())
        .with_v3_unsupported_part_behavior(opts.v3_unsupported_part_behavior);
    let bytes = encode_chat_stream_as_sse(ensure_stream_end(stream), converter);

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
    use siumai::experimental::streaming::{encode_chat_stream_as_sse, ensure_stream_end};
    use siumai::protocol::gemini::streaming::GeminiEventConverter;
    use siumai::protocol::gemini::types::GeminiConfig;

    let converter = GeminiEventConverter::new(GeminiConfig::default())
        .with_v3_unsupported_part_behavior(opts.v3_unsupported_part_behavior)
        .with_emit_function_response_tool_results(opts.gemini_emit_function_response_tool_results);
    let bytes = encode_chat_stream_as_sse(ensure_stream_end(stream), converter);

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
    report: Option<&BridgeReport>,
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
            bridge_target_for_sse(target),
            report,
            effective_bridge_mode(opts).unwrap_or(policy.bridge_options.mode),
        );
    }

    response
}

fn rejected_sse_response(
    report: BridgeReport,
    policy: Option<&GatewayBridgePolicy>,
) -> Response<Body> {
    let mut response = Response::builder()
        .status(422)
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&serde_json::json!({
                "error": "bridge rejected",
                "report": &report,
            }))
            .unwrap_or_else(|_| b"{\"error\":\"bridge rejected\"}".to_vec()),
        ))
        .unwrap_or_else(|_| Response::new(Body::from("internal error")));

    if let Some(policy) = policy {
        apply_gateway_policy_headers(
            response.headers_mut(),
            policy,
            report.target,
            Some(&report),
            report.mode,
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
    resolve_bridge_options(opts).map(|options| options.mode)
}

fn resolve_bridge_options(opts: &TranscodeSseOptions) -> Option<BridgeOptions> {
    if opts.policy.is_none()
        && opts.bridge_options.is_none()
        && opts.bridge_source.is_none()
        && opts.bridge_options_override.is_none()
    {
        return None;
    }

    Some(resolve_gateway_bridge_options(
        opts.policy.as_ref(),
        opts.bridge_options.clone(),
        opts.bridge_options_override.clone(),
    ))
}

fn apply_runtime_stream_policy(
    stream: ByteStream,
    target: TargetSseFormat,
    opts: &TranscodeSseOptions,
) -> ByteStream {
    let Some(policy) = gateway_sse_runtime_policy(opts.policy.as_ref()) else {
        return stream;
    };

    let keepalive_interval = policy.keepalive_interval;
    let idle_timeout = policy.idle_timeout;

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
    target: BridgeTarget,
    report: Option<&BridgeReport>,
    mode: siumai::experimental::bridge::BridgeMode,
) {
    for entry in gateway_bridge_headers(policy, target, report, mode) {
        if let Ok(value) = axum::http::HeaderValue::from_str(&entry.value) {
            headers.insert(entry.name, value);
        }
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

    #[cfg(feature = "anthropic")]
    use siumai::experimental::bridge::{
        BridgeLossAction, BridgeLossPolicy, RequestBridgeContext, ResponseBridgeContext,
        StreamBridgeContext,
    };
    use siumai::experimental::bridge::{BridgeMode, BridgeOptions, BridgeTarget};

    use crate::bridge::{ClosureBridgeCustomization, stream_bridge_hook};

    #[cfg(feature = "anthropic")]
    struct ContinueLossyPolicy;

    #[cfg(feature = "anthropic")]
    impl BridgeLossPolicy for ContinueLossyPolicy {
        fn request_action(
            &self,
            _ctx: &RequestBridgeContext,
            _report: &BridgeReport,
        ) -> BridgeLossAction {
            BridgeLossAction::Continue
        }

        fn response_action(
            &self,
            _ctx: &ResponseBridgeContext,
            _report: &BridgeReport,
        ) -> BridgeLossAction {
            BridgeLossAction::Continue
        }

        fn stream_action(
            &self,
            _ctx: &StreamBridgeContext,
            _report: &BridgeReport,
        ) -> BridgeLossAction {
            BridgeLossAction::Continue
        }
    }

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
    async fn transcode_sse_bridge_customization_can_transform_events() {
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
            TranscodeSseOptions::default().with_bridge_customization(Arc::new(
                ClosureBridgeCustomization::default().with_stream(|ctx, event| {
                    assert_eq!(ctx.target, BridgeTarget::OpenAiResponses);
                    match event {
                        ChatStreamEvent::ContentDelta { delta, index } => {
                            vec![ChatStreamEvent::ContentDelta {
                                delta: delta.to_uppercase(),
                                index,
                            }]
                        }
                        other => vec![other],
                    }
                }),
            )),
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
    async fn transcode_sse_route_mode_override_updates_effective_headers() {
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
            TranscodeSseOptions::default()
                .with_policy(
                    GatewayBridgePolicy::new(BridgeMode::BestEffort).with_bridge_headers(true),
                )
                .with_bridge_mode_override(BridgeMode::Strict),
        );

        assert_eq!(resp.headers()["x-siumai-bridge-mode"], "strict");
    }

    #[tokio::test]
    #[cfg(feature = "anthropic")]
    async fn transcode_sse_strict_cross_protocol_route_rejects_when_source_is_set() {
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
            TargetSseFormat::AnthropicMessages,
            TranscodeSseOptions::default()
                .with_bridge_source(BridgeTarget::OpenAiResponses)
                .with_policy(
                    GatewayBridgePolicy::new(BridgeMode::BestEffort)
                        .with_bridge_headers(true)
                        .with_bridge_warning_headers(true),
                )
                .with_bridge_mode_override(BridgeMode::Strict),
        );

        assert_eq!(resp.status(), axum::http::StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(resp.headers()["content-type"], "application/json");
        assert_eq!(
            resp.headers()["x-siumai-bridge-target"],
            "anthropic-messages"
        );
        assert_eq!(resp.headers()["x-siumai-bridge-mode"], "strict");
        assert_eq!(resp.headers()["x-siumai-bridge-decision"], "rejected");
        assert_eq!(resp.headers()["x-siumai-bridge-lossy-fields"], "1");

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("body bytes");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(value["error"], "bridge rejected");
        assert_eq!(value["report"]["lossy_fields"][0], "stream.protocol");
    }

    #[tokio::test]
    #[cfg(feature = "anthropic")]
    async fn anthropic_direct_helper_with_options_uses_same_rejection_path() {
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

        let resp = to_anthropic_messages_sse_response_with_options(
            chat_stream,
            TranscodeSseOptions::default()
                .with_bridge_source(BridgeTarget::OpenAiResponses)
                .with_policy(
                    GatewayBridgePolicy::new(BridgeMode::BestEffort)
                        .with_bridge_headers(true)
                        .with_bridge_warning_headers(true),
                )
                .with_bridge_mode_override(BridgeMode::Strict),
        );

        assert_eq!(resp.status(), axum::http::StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(
            resp.headers()["x-siumai-bridge-target"],
            "anthropic-messages"
        );
        assert_eq!(resp.headers()["x-siumai-bridge-decision"], "rejected");
    }

    #[tokio::test]
    #[cfg(feature = "anthropic")]
    async fn transcode_sse_custom_loss_policy_can_allow_cross_protocol_strict_mode() {
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
            TargetSseFormat::AnthropicMessages,
            TranscodeSseOptions::default()
                .with_bridge_source(BridgeTarget::OpenAiResponses)
                .with_policy(
                    GatewayBridgePolicy::new(BridgeMode::BestEffort)
                        .with_bridge_headers(true)
                        .with_bridge_warning_headers(true),
                )
                .with_bridge_options_override(
                    siumai::experimental::bridge::BridgeOptionsOverride::new()
                        .with_mode(BridgeMode::Strict)
                        .with_loss_policy(Arc::new(ContinueLossyPolicy)),
                ),
        );

        assert_eq!(resp.status(), axum::http::StatusCode::OK);
        assert_eq!(
            resp.headers()["x-siumai-bridge-target"],
            "anthropic-messages"
        );
        assert_eq!(resp.headers()["x-siumai-bridge-mode"], "strict");
        assert_eq!(resp.headers()["x-siumai-bridge-decision"], "lossy");
        assert_eq!(resp.headers()["x-siumai-bridge-lossy-fields"], "1");

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("body bytes");
        let text = String::from_utf8(body.to_vec()).expect("utf8");
        assert!(text.contains("event: content_block_delta"));
        assert!(text.contains("hello"));
    }

    #[tokio::test]
    #[cfg(feature = "anthropic")]
    async fn transcode_sse_best_effort_cross_protocol_headers_show_lossy_decision() {
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
            TargetSseFormat::AnthropicMessages,
            TranscodeSseOptions::default()
                .with_bridge_source(BridgeTarget::OpenAiResponses)
                .with_policy(
                    GatewayBridgePolicy::new(BridgeMode::BestEffort)
                        .with_bridge_headers(true)
                        .with_bridge_warning_headers(true),
                ),
        );

        assert_eq!(resp.status(), axum::http::StatusCode::OK);
        assert_eq!(resp.headers()["x-siumai-bridge-decision"], "lossy");
        assert_eq!(resp.headers()["x-siumai-bridge-warnings"], "1");
        assert_eq!(resp.headers()["x-siumai-bridge-lossy-fields"], "1");
        assert_eq!(resp.headers()["x-siumai-bridge-dropped-fields"], "0");
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

    #[tokio::test]
    #[cfg(feature = "openai")]
    async fn transcode_sse_finalizes_clean_eof_without_stream_end() {
        use futures::stream;

        let chat_stream: ChatStream = Box::pin(stream::iter(vec![
            Ok(ChatStreamEvent::StreamStart {
                metadata: siumai::prelude::unified::ResponseMetadata {
                    id: Some("resp_1".to_string()),
                    model: Some("gpt-4o-mini".to_string()),
                    created: None,
                    provider: "openai".to_string(),
                    request_id: None,
                },
            }),
            Ok(ChatStreamEvent::ContentDelta {
                delta: "hello".to_string(),
                index: None,
            }),
        ]));

        let resp = to_transcoded_sse_response(
            chat_stream,
            TargetSseFormat::OpenAiResponses,
            TranscodeSseOptions::default(),
        );

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("body bytes");
        let text = String::from_utf8(body.to_vec()).expect("utf8");
        assert!(text.contains("response.completed"));
        assert!(text.contains("[DONE]"));
    }
}
