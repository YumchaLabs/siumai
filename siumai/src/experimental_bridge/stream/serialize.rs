//! Stream bridge serialization.

use futures_util::Stream;
use siumai_core::LlmError;
use siumai_core::bridge::{BridgeMode, BridgeReport, BridgeResult, BridgeTarget};
use siumai_core::streaming::{
    ChatByteStream, ChatStreamEvent, OpenAiResponsesStreamPartsBridge, encode_chat_stream_as_sse,
    transform_chat_event_stream,
};

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
    let mut report = BridgeReport::with_source(source, target, mode);
    inspect_chat_stream_bridge(source, target, &mut report);

    let stream = transform_stream_for_target(stream, source, target);
    let bytes = encode_stream_for_target(stream, target)?;

    Ok(BridgeResult::new(bytes, report))
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

fn transform_stream_for_target<S>(
    stream: S,
    source: Option<BridgeTarget>,
    target: BridgeTarget,
) -> std::pin::Pin<Box<dyn Stream<Item = Result<ChatStreamEvent, LlmError>> + Send>>
where
    S: Stream<Item = Result<ChatStreamEvent, LlmError>> + Send + 'static,
{
    if matches!(target, BridgeTarget::OpenAiResponses)
        && !matches!(source, Some(BridgeTarget::OpenAiResponses))
    {
        let mut bridge = OpenAiResponsesStreamPartsBridge::new();
        return transform_chat_event_stream(stream, move |event| bridge.bridge_event(event));
    }

    Box::pin(stream)
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
