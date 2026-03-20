//! Stream bridge serialization.

use futures_util::Stream;
use siumai_core::LlmError;
use siumai_core::bridge::{
    BridgeLossAction, BridgeMode, BridgeOptions, BridgeReport, BridgeResult, BridgeTarget,
    StreamBridgeContext,
};
use siumai_core::streaming::{
    ChatByteStream, ChatStreamEvent, OpenAiResponsesStreamPartsBridge, encode_chat_stream_as_sse,
    transform_chat_event_stream,
};

use crate::experimental_bridge::customize::remap_stream_event;

use super::inspect::inspect_chat_stream_bridge;

/// Bridge a normalized `ChatStreamEvent` stream into a target protocol byte stream.
pub fn bridge_chat_stream_to_bytes<S>(
    stream: S,
    source: Option<BridgeTarget>,
    target: BridgeTarget,
    mode: BridgeMode,
) -> Result<BridgeResult<ChatByteStream>, LlmError>
where
    S: Stream<Item = Result<ChatStreamEvent, LlmError>> + Send + 'static,
{
    bridge_chat_stream_to_bytes_with_options(stream, source, target, BridgeOptions::new(mode))
}

/// Convenience wrapper for `OpenAI Responses`.
#[cfg(feature = "openai")]
pub fn bridge_chat_stream_to_openai_responses_sse<S>(
    stream: S,
    source: Option<BridgeTarget>,
    mode: BridgeMode,
) -> Result<BridgeResult<ChatByteStream>, LlmError>
where
    S: Stream<Item = Result<ChatStreamEvent, LlmError>> + Send + 'static,
{
    bridge_chat_stream_to_bytes(stream, source, BridgeTarget::OpenAiResponses, mode)
}

/// Convenience wrapper for `OpenAI Responses` with bridge customization.
#[cfg(feature = "openai")]
pub fn bridge_chat_stream_to_openai_responses_sse_with_options<S>(
    stream: S,
    source: Option<BridgeTarget>,
    options: BridgeOptions,
) -> Result<BridgeResult<ChatByteStream>, LlmError>
where
    S: Stream<Item = Result<ChatStreamEvent, LlmError>> + Send + 'static,
{
    bridge_chat_stream_to_bytes_with_options(stream, source, BridgeTarget::OpenAiResponses, options)
}

/// Convenience wrapper for `OpenAI Chat Completions`.
#[cfg(feature = "openai")]
pub fn bridge_chat_stream_to_openai_chat_completions_sse<S>(
    stream: S,
    source: Option<BridgeTarget>,
    mode: BridgeMode,
) -> Result<BridgeResult<ChatByteStream>, LlmError>
where
    S: Stream<Item = Result<ChatStreamEvent, LlmError>> + Send + 'static,
{
    bridge_chat_stream_to_bytes(stream, source, BridgeTarget::OpenAiChatCompletions, mode)
}

/// Convenience wrapper for `OpenAI Chat Completions` with bridge customization.
#[cfg(feature = "openai")]
pub fn bridge_chat_stream_to_openai_chat_completions_sse_with_options<S>(
    stream: S,
    source: Option<BridgeTarget>,
    options: BridgeOptions,
) -> Result<BridgeResult<ChatByteStream>, LlmError>
where
    S: Stream<Item = Result<ChatStreamEvent, LlmError>> + Send + 'static,
{
    bridge_chat_stream_to_bytes_with_options(
        stream,
        source,
        BridgeTarget::OpenAiChatCompletions,
        options,
    )
}

/// Convenience wrapper for `Anthropic Messages`.
#[cfg(feature = "anthropic")]
pub fn bridge_chat_stream_to_anthropic_messages_sse<S>(
    stream: S,
    source: Option<BridgeTarget>,
    mode: BridgeMode,
) -> Result<BridgeResult<ChatByteStream>, LlmError>
where
    S: Stream<Item = Result<ChatStreamEvent, LlmError>> + Send + 'static,
{
    bridge_chat_stream_to_bytes(stream, source, BridgeTarget::AnthropicMessages, mode)
}

/// Convenience wrapper for `Anthropic Messages` with bridge customization.
#[cfg(feature = "anthropic")]
pub fn bridge_chat_stream_to_anthropic_messages_sse_with_options<S>(
    stream: S,
    source: Option<BridgeTarget>,
    options: BridgeOptions,
) -> Result<BridgeResult<ChatByteStream>, LlmError>
where
    S: Stream<Item = Result<ChatStreamEvent, LlmError>> + Send + 'static,
{
    bridge_chat_stream_to_bytes_with_options(
        stream,
        source,
        BridgeTarget::AnthropicMessages,
        options,
    )
}

/// Convenience wrapper for `Gemini GenerateContent`.
#[cfg(feature = "google")]
pub fn bridge_chat_stream_to_gemini_generate_content_sse<S>(
    stream: S,
    source: Option<BridgeTarget>,
    mode: BridgeMode,
) -> Result<BridgeResult<ChatByteStream>, LlmError>
where
    S: Stream<Item = Result<ChatStreamEvent, LlmError>> + Send + 'static,
{
    bridge_chat_stream_to_bytes(stream, source, BridgeTarget::GeminiGenerateContent, mode)
}

/// Convenience wrapper for `Gemini GenerateContent` with bridge customization.
#[cfg(feature = "google")]
pub fn bridge_chat_stream_to_gemini_generate_content_sse_with_options<S>(
    stream: S,
    source: Option<BridgeTarget>,
    options: BridgeOptions,
) -> Result<BridgeResult<ChatByteStream>, LlmError>
where
    S: Stream<Item = Result<ChatStreamEvent, LlmError>> + Send + 'static,
{
    bridge_chat_stream_to_bytes_with_options(
        stream,
        source,
        BridgeTarget::GeminiGenerateContent,
        options,
    )
}

/// Bridge a normalized `ChatStreamEvent` stream into a target protocol byte stream with bridge
/// customization.
pub fn bridge_chat_stream_to_bytes_with_options<S>(
    stream: S,
    source: Option<BridgeTarget>,
    target: BridgeTarget,
    options: BridgeOptions,
) -> Result<BridgeResult<ChatByteStream>, LlmError>
where
    S: Stream<Item = Result<ChatStreamEvent, LlmError>> + Send + 'static,
{
    let ctx = StreamBridgeContext::new(
        source,
        target,
        options.mode,
        options.route_label.clone(),
        Some(stream_path_label(source, target).to_string()),
    );
    let mut report = BridgeReport::with_source(source, target, options.mode);
    inspect_chat_stream_bridge(source, target, &mut report);

    if should_reject_stream(&options, &ctx, &mut report) {
        return Ok(BridgeResult::rejected(report));
    }

    let stream = transform_stream_for_target(stream, source, target, &ctx, &options);
    let bytes = encode_stream_for_target(stream, target)?;

    Ok(BridgeResult::new(bytes, report))
}

/// Apply bridge stream hooks and primitive remappers without encoding into a target wire format.
///
/// This is useful for gateway adapters that still need target-specific serializer options
/// outside the core bridge surface.
pub fn transform_chat_stream_with_bridge_options<S>(
    stream: S,
    source: Option<BridgeTarget>,
    target: BridgeTarget,
    options: &BridgeOptions,
    path_label: Option<String>,
) -> std::pin::Pin<Box<dyn Stream<Item = Result<ChatStreamEvent, LlmError>> + Send>>
where
    S: Stream<Item = Result<ChatStreamEvent, LlmError>> + Send + 'static,
{
    if options.primitive_remapper.is_none() && options.stream_hook.is_none() {
        return Box::pin(stream);
    }

    let remapper = options.primitive_remapper.clone();
    let hook = options.stream_hook.clone();
    let ctx = StreamBridgeContext::new(
        source,
        target,
        options.mode,
        options.route_label.clone(),
        path_label,
    );

    transform_chat_event_stream(stream, move |event| {
        let mut events = if let Some(remapper) = remapper.as_deref() {
            vec![remap_stream_event(event, &ctx, remapper)]
        } else {
            vec![event]
        };

        if let Some(hook) = hook.as_deref() {
            let mut mapped = Vec::new();
            for event in events.drain(..) {
                mapped.extend(hook.map_event(&ctx, event));
            }
            mapped
        } else {
            events
        }
    })
}

fn transform_stream_for_target<S>(
    stream: S,
    source: Option<BridgeTarget>,
    target: BridgeTarget,
    ctx: &StreamBridgeContext,
    options: &BridgeOptions,
) -> std::pin::Pin<Box<dyn Stream<Item = Result<ChatStreamEvent, LlmError>> + Send>>
where
    S: Stream<Item = Result<ChatStreamEvent, LlmError>> + Send + 'static,
{
    let stream: std::pin::Pin<Box<dyn Stream<Item = Result<ChatStreamEvent, LlmError>> + Send>> =
        if matches!(target, BridgeTarget::OpenAiResponses)
            && !matches!(source, Some(BridgeTarget::OpenAiResponses))
        {
            let mut bridge = OpenAiResponsesStreamPartsBridge::new();
            transform_chat_event_stream(stream, move |event| bridge.bridge_event(event))
        } else {
            Box::pin(stream)
        };

    transform_chat_stream_with_bridge_options(
        stream,
        ctx.source,
        ctx.target,
        options,
        ctx.path_label.clone(),
    )
}

fn stream_path_label(source: Option<BridgeTarget>, target: BridgeTarget) -> &'static str {
    if matches!(target, BridgeTarget::OpenAiResponses)
        && !matches!(source, Some(BridgeTarget::OpenAiResponses))
    {
        "openai-responses-stream-adapter"
    } else {
        "protocol-event-serialization"
    }
}

fn should_reject_stream(
    options: &BridgeOptions,
    ctx: &StreamBridgeContext,
    report: &mut BridgeReport,
) -> bool {
    if report.is_rejected() {
        return true;
    }

    if matches!(
        options.loss_policy.stream_action(ctx, report),
        BridgeLossAction::Reject
    ) {
        report.reject(format!(
            "bridge policy rejected stream conversion to {}",
            ctx.target.as_str()
        ));
        return true;
    }

    false
}

fn encode_stream_for_target<S>(stream: S, target: BridgeTarget) -> Result<ChatByteStream, LlmError>
where
    S: Stream<Item = Result<ChatStreamEvent, LlmError>> + Send + 'static,
{
    match target {
        BridgeTarget::OpenAiResponses => {
            #[cfg(feature = "openai")]
            {
                Ok(encode_chat_stream_as_sse(
                    stream,
                    siumai_protocol_openai::standards::openai::responses_sse::OpenAiResponsesEventConverter::new(),
                ))
            }
            #[cfg(not(feature = "openai"))]
            {
                Err(LlmError::UnsupportedOperation(
                    "openai feature is disabled".to_string(),
                ))
            }
        }
        BridgeTarget::OpenAiChatCompletions => {
            #[cfg(feature = "openai")]
            {
                let adapter = std::sync::Arc::new(
                    siumai_core::standards::openai::compat::adapter::OpenAiStandardAdapter {
                        base_url: String::new(),
                    },
                );
                let cfg = siumai_core::standards::openai::compat::openai_config::OpenAiCompatibleConfig::new(
                    "openai",
                    "",
                    "",
                    adapter.clone(),
                );
                Ok(encode_chat_stream_as_sse(
                    stream,
                    siumai_core::standards::openai::compat::streaming::OpenAiCompatibleEventConverter::new(
                        cfg,
                        adapter,
                    ),
                ))
            }
            #[cfg(not(feature = "openai"))]
            {
                Err(LlmError::UnsupportedOperation(
                    "openai feature is disabled".to_string(),
                ))
            }
        }
        BridgeTarget::AnthropicMessages => {
            #[cfg(feature = "anthropic")]
            {
                Ok(encode_chat_stream_as_sse(
                    stream,
                    siumai_protocol_anthropic::standards::anthropic::streaming::AnthropicEventConverter::new(
                        siumai_protocol_anthropic::standards::anthropic::params::AnthropicParams::default(),
                    ),
                ))
            }
            #[cfg(not(feature = "anthropic"))]
            {
                Err(LlmError::UnsupportedOperation(
                    "anthropic feature is disabled".to_string(),
                ))
            }
        }
        BridgeTarget::GeminiGenerateContent => {
            #[cfg(feature = "google")]
            {
                Ok(encode_chat_stream_as_sse(
                    stream,
                    siumai_protocol_gemini::standards::gemini::streaming::GeminiEventConverter::new(
                        siumai_protocol_gemini::standards::gemini::types::GeminiConfig::default(),
                    ),
                ))
            }
            #[cfg(not(feature = "google"))]
            {
                Err(LlmError::UnsupportedOperation(
                    "google feature is disabled".to_string(),
                ))
            }
        }
    }
}
