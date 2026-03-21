//! Response bridge serialization.

use siumai_core::LlmError;
use siumai_core::bridge::{BridgeMode, BridgeOptions, BridgeResult, BridgeTarget};
use siumai_core::encoding::JsonEncodeOptions;
use siumai_core::types::ChatResponse;

use crate::experimental_bridge::customize::apply_response_remapper;
use crate::experimental_bridge::lifecycle::{
    NORMALIZED_RESPONSE_PATH_LABEL, new_bridge_report, new_response_context, reject_if_needed,
};
use crate::experimental_bridge::target_dispatch::encode_chat_response_to_json_bytes;
use crate::experimental_bridge::wrapper_macros::define_response_bridge_wrappers;

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
    let ctx = new_response_context(
        source,
        target,
        &options,
        Some(NORMALIZED_RESPONSE_PATH_LABEL.to_string()),
    );
    let mut report = new_bridge_report(source, target, options.mode);
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

    let bytes = encode_chat_response_to_json_bytes(&working_response, target, opts)?;

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

define_response_bridge_wrappers!(
    "openai",
    bridge_chat_response_to_openai_responses_json_bytes,
    bridge_chat_response_to_openai_responses_json_bytes_with_options,
    bridge_chat_response_to_openai_responses_json_value,
    bridge_chat_response_to_openai_responses_json_value_with_options,
    BridgeTarget::OpenAiResponses,
    "OpenAI Responses"
);
define_response_bridge_wrappers!(
    "openai",
    bridge_chat_response_to_openai_chat_completions_json_bytes,
    bridge_chat_response_to_openai_chat_completions_json_bytes_with_options,
    bridge_chat_response_to_openai_chat_completions_json_value,
    bridge_chat_response_to_openai_chat_completions_json_value_with_options,
    BridgeTarget::OpenAiChatCompletions,
    "OpenAI Chat Completions"
);
define_response_bridge_wrappers!(
    "anthropic",
    bridge_chat_response_to_anthropic_messages_json_bytes,
    bridge_chat_response_to_anthropic_messages_json_bytes_with_options,
    bridge_chat_response_to_anthropic_messages_json_value,
    bridge_chat_response_to_anthropic_messages_json_value_with_options,
    BridgeTarget::AnthropicMessages,
    "Anthropic Messages"
);
define_response_bridge_wrappers!(
    "google",
    bridge_chat_response_to_gemini_generate_content_json_bytes,
    bridge_chat_response_to_gemini_generate_content_json_bytes_with_options,
    bridge_chat_response_to_gemini_generate_content_json_value,
    bridge_chat_response_to_gemini_generate_content_json_value_with_options,
    BridgeTarget::GeminiGenerateContent,
    "Gemini GenerateContent"
);

fn should_reject_response(
    options: &BridgeOptions,
    ctx: &siumai_core::bridge::ResponseBridgeContext,
    report: &mut siumai_core::bridge::BridgeReport,
) -> bool {
    let action = options.loss_policy.response_action(ctx, report);
    reject_if_needed(report, action, "response", ctx.target)
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
