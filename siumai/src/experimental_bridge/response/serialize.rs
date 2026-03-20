//! Response bridge serialization.

use siumai_core::LlmError;
use siumai_core::bridge::{
    BridgeLossAction, BridgeMode, BridgeOptions, BridgeReport, BridgeResult, BridgeTarget,
    ResponseBridgeContext,
};
use siumai_core::encoding::{JsonEncodeOptions, encode_chat_response_as_json};
use siumai_core::types::ChatResponse;

use crate::experimental_bridge::customize::apply_response_remapper;

use super::inspect::inspect_chat_response_bridge;

/// Bridge a normalized `ChatResponse` into a target protocol JSON body.
pub fn bridge_chat_response_to_json_bytes(
    response: &ChatResponse,
    source: Option<BridgeTarget>,
    target: BridgeTarget,
    mode: BridgeMode,
    opts: JsonEncodeOptions,
) -> Result<BridgeResult<Vec<u8>>, LlmError> {
    bridge_chat_response_to_json_bytes_with_options(
        response,
        source,
        target,
        BridgeOptions::new(mode),
        opts,
    )
}

/// Bridge a normalized `ChatResponse` into a target protocol JSON body with bridge customization.
pub fn bridge_chat_response_to_json_bytes_with_options(
    response: &ChatResponse,
    source: Option<BridgeTarget>,
    target: BridgeTarget,
    options: BridgeOptions,
    opts: JsonEncodeOptions,
) -> Result<BridgeResult<Vec<u8>>, LlmError> {
    let ctx = ResponseBridgeContext::new(
        source,
        target,
        options.mode,
        options.route_label.clone(),
        Some("normalized-response".to_string()),
    );
    let mut report = BridgeReport::with_source(source, target, options.mode);
    let mut working_response = response.clone();

    if let Some(remapper) = options.primitive_remapper.as_deref() {
        apply_response_remapper(&mut working_response, &ctx, remapper);
    }
    if let Some(hook) = options.response_hook.as_deref() {
        hook.transform_response(&ctx, &mut working_response, &mut report)?;
    }

    inspect_chat_response_bridge(&working_response, target, &mut report);

    if should_reject_response(&options, &ctx, &mut report) {
        return Ok(BridgeResult::rejected(report));
    }

    let bytes = transform_chat_response_to_json_bytes(&working_response, target, opts)?;

    if should_reject_response(&options, &ctx, &mut report) {
        return Ok(BridgeResult::rejected(report));
    }

    Ok(BridgeResult::new(bytes, report))
}

/// Bridge a normalized `ChatResponse` into a target protocol JSON value.
pub fn bridge_chat_response_to_json_value(
    response: &ChatResponse,
    source: Option<BridgeTarget>,
    target: BridgeTarget,
    mode: BridgeMode,
    opts: JsonEncodeOptions,
) -> Result<BridgeResult<serde_json::Value>, LlmError> {
    let result = bridge_chat_response_to_json_bytes(response, source, target, mode, opts)?;
    result.map_or_json_value()
}

/// Bridge a normalized `ChatResponse` into a target protocol JSON value with bridge customization.
pub fn bridge_chat_response_to_json_value_with_options(
    response: &ChatResponse,
    source: Option<BridgeTarget>,
    target: BridgeTarget,
    options: BridgeOptions,
    opts: JsonEncodeOptions,
) -> Result<BridgeResult<serde_json::Value>, LlmError> {
    let result =
        bridge_chat_response_to_json_bytes_with_options(response, source, target, options, opts)?;
    result.map_or_json_value()
}

/// Convenience wrapper for `OpenAI Responses`.
#[cfg(feature = "openai")]
pub fn bridge_chat_response_to_openai_responses_json_bytes(
    response: &ChatResponse,
    source: Option<BridgeTarget>,
    mode: BridgeMode,
    opts: JsonEncodeOptions,
) -> Result<BridgeResult<Vec<u8>>, LlmError> {
    bridge_chat_response_to_json_bytes(response, source, BridgeTarget::OpenAiResponses, mode, opts)
}

/// Convenience wrapper for `OpenAI Responses` with bridge customization.
#[cfg(feature = "openai")]
pub fn bridge_chat_response_to_openai_responses_json_bytes_with_options(
    response: &ChatResponse,
    source: Option<BridgeTarget>,
    options: BridgeOptions,
    opts: JsonEncodeOptions,
) -> Result<BridgeResult<Vec<u8>>, LlmError> {
    bridge_chat_response_to_json_bytes_with_options(
        response,
        source,
        BridgeTarget::OpenAiResponses,
        options,
        opts,
    )
}

/// Convenience wrapper for `OpenAI Responses`.
#[cfg(feature = "openai")]
pub fn bridge_chat_response_to_openai_responses_json_value(
    response: &ChatResponse,
    source: Option<BridgeTarget>,
    mode: BridgeMode,
    opts: JsonEncodeOptions,
) -> Result<BridgeResult<serde_json::Value>, LlmError> {
    bridge_chat_response_to_json_value(response, source, BridgeTarget::OpenAiResponses, mode, opts)
}

/// Convenience wrapper for `OpenAI Responses` with bridge customization.
#[cfg(feature = "openai")]
pub fn bridge_chat_response_to_openai_responses_json_value_with_options(
    response: &ChatResponse,
    source: Option<BridgeTarget>,
    options: BridgeOptions,
    opts: JsonEncodeOptions,
) -> Result<BridgeResult<serde_json::Value>, LlmError> {
    bridge_chat_response_to_json_value_with_options(
        response,
        source,
        BridgeTarget::OpenAiResponses,
        options,
        opts,
    )
}

/// Convenience wrapper for `OpenAI Chat Completions`.
#[cfg(feature = "openai")]
pub fn bridge_chat_response_to_openai_chat_completions_json_bytes(
    response: &ChatResponse,
    source: Option<BridgeTarget>,
    mode: BridgeMode,
    opts: JsonEncodeOptions,
) -> Result<BridgeResult<Vec<u8>>, LlmError> {
    bridge_chat_response_to_json_bytes(
        response,
        source,
        BridgeTarget::OpenAiChatCompletions,
        mode,
        opts,
    )
}

/// Convenience wrapper for `OpenAI Chat Completions` with bridge customization.
#[cfg(feature = "openai")]
pub fn bridge_chat_response_to_openai_chat_completions_json_bytes_with_options(
    response: &ChatResponse,
    source: Option<BridgeTarget>,
    options: BridgeOptions,
    opts: JsonEncodeOptions,
) -> Result<BridgeResult<Vec<u8>>, LlmError> {
    bridge_chat_response_to_json_bytes_with_options(
        response,
        source,
        BridgeTarget::OpenAiChatCompletions,
        options,
        opts,
    )
}

/// Convenience wrapper for `OpenAI Chat Completions`.
#[cfg(feature = "openai")]
pub fn bridge_chat_response_to_openai_chat_completions_json_value(
    response: &ChatResponse,
    source: Option<BridgeTarget>,
    mode: BridgeMode,
    opts: JsonEncodeOptions,
) -> Result<BridgeResult<serde_json::Value>, LlmError> {
    bridge_chat_response_to_json_value(
        response,
        source,
        BridgeTarget::OpenAiChatCompletions,
        mode,
        opts,
    )
}

/// Convenience wrapper for `OpenAI Chat Completions` with bridge customization.
#[cfg(feature = "openai")]
pub fn bridge_chat_response_to_openai_chat_completions_json_value_with_options(
    response: &ChatResponse,
    source: Option<BridgeTarget>,
    options: BridgeOptions,
    opts: JsonEncodeOptions,
) -> Result<BridgeResult<serde_json::Value>, LlmError> {
    bridge_chat_response_to_json_value_with_options(
        response,
        source,
        BridgeTarget::OpenAiChatCompletions,
        options,
        opts,
    )
}

/// Convenience wrapper for `Anthropic Messages`.
#[cfg(feature = "anthropic")]
pub fn bridge_chat_response_to_anthropic_messages_json_bytes(
    response: &ChatResponse,
    source: Option<BridgeTarget>,
    mode: BridgeMode,
    opts: JsonEncodeOptions,
) -> Result<BridgeResult<Vec<u8>>, LlmError> {
    bridge_chat_response_to_json_bytes(
        response,
        source,
        BridgeTarget::AnthropicMessages,
        mode,
        opts,
    )
}

/// Convenience wrapper for `Anthropic Messages` with bridge customization.
#[cfg(feature = "anthropic")]
pub fn bridge_chat_response_to_anthropic_messages_json_bytes_with_options(
    response: &ChatResponse,
    source: Option<BridgeTarget>,
    options: BridgeOptions,
    opts: JsonEncodeOptions,
) -> Result<BridgeResult<Vec<u8>>, LlmError> {
    bridge_chat_response_to_json_bytes_with_options(
        response,
        source,
        BridgeTarget::AnthropicMessages,
        options,
        opts,
    )
}

/// Convenience wrapper for `Anthropic Messages`.
#[cfg(feature = "anthropic")]
pub fn bridge_chat_response_to_anthropic_messages_json_value(
    response: &ChatResponse,
    source: Option<BridgeTarget>,
    mode: BridgeMode,
    opts: JsonEncodeOptions,
) -> Result<BridgeResult<serde_json::Value>, LlmError> {
    bridge_chat_response_to_json_value(
        response,
        source,
        BridgeTarget::AnthropicMessages,
        mode,
        opts,
    )
}

/// Convenience wrapper for `Anthropic Messages` with bridge customization.
#[cfg(feature = "anthropic")]
pub fn bridge_chat_response_to_anthropic_messages_json_value_with_options(
    response: &ChatResponse,
    source: Option<BridgeTarget>,
    options: BridgeOptions,
    opts: JsonEncodeOptions,
) -> Result<BridgeResult<serde_json::Value>, LlmError> {
    bridge_chat_response_to_json_value_with_options(
        response,
        source,
        BridgeTarget::AnthropicMessages,
        options,
        opts,
    )
}

/// Convenience wrapper for `Gemini GenerateContent`.
#[cfg(feature = "google")]
pub fn bridge_chat_response_to_gemini_generate_content_json_bytes(
    response: &ChatResponse,
    source: Option<BridgeTarget>,
    mode: BridgeMode,
    opts: JsonEncodeOptions,
) -> Result<BridgeResult<Vec<u8>>, LlmError> {
    bridge_chat_response_to_json_bytes(
        response,
        source,
        BridgeTarget::GeminiGenerateContent,
        mode,
        opts,
    )
}

/// Convenience wrapper for `Gemini GenerateContent` with bridge customization.
#[cfg(feature = "google")]
pub fn bridge_chat_response_to_gemini_generate_content_json_bytes_with_options(
    response: &ChatResponse,
    source: Option<BridgeTarget>,
    options: BridgeOptions,
    opts: JsonEncodeOptions,
) -> Result<BridgeResult<Vec<u8>>, LlmError> {
    bridge_chat_response_to_json_bytes_with_options(
        response,
        source,
        BridgeTarget::GeminiGenerateContent,
        options,
        opts,
    )
}

/// Convenience wrapper for `Gemini GenerateContent`.
#[cfg(feature = "google")]
pub fn bridge_chat_response_to_gemini_generate_content_json_value(
    response: &ChatResponse,
    source: Option<BridgeTarget>,
    mode: BridgeMode,
    opts: JsonEncodeOptions,
) -> Result<BridgeResult<serde_json::Value>, LlmError> {
    bridge_chat_response_to_json_value(
        response,
        source,
        BridgeTarget::GeminiGenerateContent,
        mode,
        opts,
    )
}

/// Convenience wrapper for `Gemini GenerateContent` with bridge customization.
#[cfg(feature = "google")]
pub fn bridge_chat_response_to_gemini_generate_content_json_value_with_options(
    response: &ChatResponse,
    source: Option<BridgeTarget>,
    options: BridgeOptions,
    opts: JsonEncodeOptions,
) -> Result<BridgeResult<serde_json::Value>, LlmError> {
    bridge_chat_response_to_json_value_with_options(
        response,
        source,
        BridgeTarget::GeminiGenerateContent,
        options,
        opts,
    )
}

fn should_reject_response(
    options: &BridgeOptions,
    ctx: &ResponseBridgeContext,
    report: &mut BridgeReport,
) -> bool {
    if report.is_rejected() {
        return true;
    }

    if matches!(
        options.loss_policy.response_action(ctx, report),
        BridgeLossAction::Reject
    ) {
        report.reject(format!(
            "bridge policy rejected response conversion to {}",
            ctx.target.as_str()
        ));
        return true;
    }

    false
}

fn transform_chat_response_to_json_bytes(
    response: &ChatResponse,
    target: BridgeTarget,
    opts: JsonEncodeOptions,
) -> Result<Vec<u8>, LlmError> {
    match target {
        BridgeTarget::OpenAiResponses => {
            #[cfg(feature = "openai")]
            {
                encode_chat_response_as_json(
                    response,
                    siumai_protocol_openai::standards::openai::json_response::OpenAiResponsesJsonResponseConverter::new(),
                    opts,
                )
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
                encode_chat_response_as_json(
                    response,
                    siumai_protocol_openai::standards::openai::json_response::OpenAiChatCompletionsJsonResponseConverter::new(),
                    opts,
                )
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
                encode_chat_response_as_json(
                    response,
                    siumai_protocol_anthropic::standards::anthropic::json_response::AnthropicMessagesJsonResponseConverter::new(),
                    opts,
                )
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
                encode_chat_response_as_json(
                    response,
                    siumai_protocol_gemini::standards::gemini::json_response::GeminiGenerateContentJsonResponseConverter::new(),
                    opts,
                )
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

trait BridgeResultJsonExt {
    fn map_or_json_value(self) -> Result<BridgeResult<serde_json::Value>, LlmError>;
}

impl BridgeResultJsonExt for BridgeResult<Vec<u8>> {
    fn map_or_json_value(self) -> Result<BridgeResult<serde_json::Value>, LlmError> {
        let BridgeResult { value, report } = self;

        match value {
            Some(bytes) => {
                let value =
                    serde_json::from_slice::<serde_json::Value>(&bytes).map_err(|error| {
                        LlmError::JsonError(format!(
                            "Failed to parse bridged response JSON bytes: {error}"
                        ))
                    })?;
                Ok(BridgeResult::new(value, report))
            }
            None => Ok(BridgeResult::rejected(report)),
        }
    }
}
