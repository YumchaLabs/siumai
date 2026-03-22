//! Stream bridge serialization.

use futures_util::Stream;
use siumai_core::LlmError;
use siumai_core::bridge::{BridgeMode, BridgeOptions, BridgeResult, BridgeTarget};
use siumai_core::streaming::{
    ChatByteStream, ChatStreamEvent, OpenAiResponsesStreamPartsBridge, ensure_stream_end,
    transform_chat_event_stream,
};

use crate::experimental_bridge::customize::remap_stream_event;
use crate::experimental_bridge::lifecycle::{
    new_bridge_report, new_stream_context, reject_if_needed,
};
use crate::experimental_bridge::stream::profile::stream_bridge_profile;
use crate::experimental_bridge::target_dispatch::encode_chat_stream_for_target;
use crate::experimental_bridge::wrapper_macros::define_stream_bridge_wrappers;

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

define_stream_bridge_wrappers!(
    feature = "openai",
    bridge_chat_stream_to_openai_responses_sse,
    bridge_chat_stream_to_openai_responses_sse_with_options,
    BridgeTarget::OpenAiResponses,
    "OpenAI Responses"
);
define_stream_bridge_wrappers!(
    feature = "openai",
    bridge_chat_stream_to_openai_chat_completions_sse,
    bridge_chat_stream_to_openai_chat_completions_sse_with_options,
    BridgeTarget::OpenAiChatCompletions,
    "OpenAI Chat Completions"
);
define_stream_bridge_wrappers!(
    feature = "anthropic",
    bridge_chat_stream_to_anthropic_messages_sse,
    bridge_chat_stream_to_anthropic_messages_sse_with_options,
    BridgeTarget::AnthropicMessages,
    "Anthropic Messages"
);
define_stream_bridge_wrappers!(
    any(feature = "google", feature = "google-vertex"),
    bridge_chat_stream_to_gemini_generate_content_sse,
    bridge_chat_stream_to_gemini_generate_content_sse_with_options,
    BridgeTarget::GeminiGenerateContent,
    "Gemini GenerateContent"
);

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
    let profile = stream_bridge_profile(source, target);
    let ctx = new_stream_context(
        source,
        target,
        &options,
        Some(profile.path_label.to_string()),
    );
    let mut report = new_bridge_report(source, target, options.mode);
    inspect_chat_stream_bridge(source, target, &mut report);

    if should_reject_stream(&options, &ctx, &mut report) {
        return Ok(BridgeResult::rejected(report));
    }

    let stream = ensure_stream_end(transform_stream_for_target(
        stream, source, target, &ctx, &options,
    ));
    let bytes = encode_chat_stream_for_target(stream, target)?;

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
    let ctx = new_stream_context(source, target, options, path_label);

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
    ctx: &siumai_core::bridge::StreamBridgeContext,
    options: &BridgeOptions,
) -> std::pin::Pin<Box<dyn Stream<Item = Result<ChatStreamEvent, LlmError>> + Send>>
where
    S: Stream<Item = Result<ChatStreamEvent, LlmError>> + Send + 'static,
{
    let stream: std::pin::Pin<Box<dyn Stream<Item = Result<ChatStreamEvent, LlmError>> + Send>> =
        if stream_bridge_profile(source, target).requires_openai_responses_stream_adapter {
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

fn should_reject_stream(
    options: &BridgeOptions,
    ctx: &siumai_core::bridge::StreamBridgeContext,
    report: &mut siumai_core::bridge::BridgeReport,
) -> bool {
    let action = options.loss_policy.stream_action(ctx, report);
    reject_if_needed(report, action, "stream", ctx.target)
}
