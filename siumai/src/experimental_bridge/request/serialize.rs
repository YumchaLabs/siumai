//! Request bridge serialization.

use crate::experimental_bridge::customize::apply_request_remapper;
use crate::experimental_bridge::lifecycle::{
    new_bridge_report, new_request_context, reject_if_needed, request_path_label,
};
use crate::experimental_bridge::planner::{RequestBridgePath, plan_chat_request_bridge};
use crate::experimental_bridge::target_dispatch::transform_chat_request_to_json;
use crate::experimental_bridge::wrapper_macros::define_request_bridge_wrappers;
use siumai_core::LlmError;
use siumai_core::bridge::{BridgeMode, BridgeOptions, BridgeResult, BridgeTarget};
use siumai_core::types::ChatRequest;

use super::inspect::inspect_chat_request_bridge;
use super::pairs::serialize_direct_request_bridge_pair;

/// Bridge a normalized `ChatRequest` into a target protocol JSON body.
///
/// The returned report is conservative: it records known lossy or dropped
/// semantics before the target transformer runs. In `Strict` mode, any lossy
/// condition rejects the bridge before serialization.
pub fn bridge_chat_request_to_json(
    request: &ChatRequest,
    source: Option<BridgeTarget>,
    target: BridgeTarget,
    mode: BridgeMode,
) -> Result<BridgeResult<serde_json::Value>, LlmError> {
    bridge_chat_request_to_json_with_options(request, source, target, BridgeOptions::new(mode))
}

/// Bridge a normalized `ChatRequest` into a target protocol JSON body with bridge customization.
pub fn bridge_chat_request_to_json_with_options(
    request: &ChatRequest,
    source: Option<BridgeTarget>,
    target: BridgeTarget,
    options: BridgeOptions,
) -> Result<BridgeResult<serde_json::Value>, LlmError> {
    let plan = plan_chat_request_bridge(source, target);
    let ctx = new_request_context(source, target, &options, Some(request_path_label(&plan)));
    let mut report = new_bridge_report(source, target, options.mode);
    let mut working_request = request.clone();

    if let Some(remapper) = options.primitive_remapper.as_deref() {
        apply_request_remapper(&mut working_request, &ctx, remapper);
    }
    if let Some(hook) = options.request_hook.as_deref() {
        hook.transform_request(&ctx, &mut working_request, &mut report)?;
    }

    inspect_chat_request_bridge(&working_request, target, &mut report);

    if should_reject_request(&options, &ctx, &mut report) {
        return Ok(BridgeResult::rejected(report));
    }

    let mut value = match plan.path {
        RequestBridgePath::ViaNormalized => {
            transform_chat_request_to_json(&working_request, target)?
        }
        RequestBridgePath::Direct(pair) => {
            serialize_direct_request_bridge_pair(pair, &working_request, &mut report)?
        }
    };

    if let Some(hook) = options.request_hook.as_deref() {
        hook.transform_json(&ctx, &mut value, &mut report)?;
        hook.validate_json(&ctx, &value, &mut report)?;
    }

    if should_reject_request(&options, &ctx, &mut report) {
        return Ok(BridgeResult::rejected(report));
    }

    Ok(BridgeResult::new(value, report))
}

define_request_bridge_wrappers!(
    feature = "openai",
    bridge_chat_request_to_openai_responses_json,
    bridge_chat_request_to_openai_responses_json_with_options,
    BridgeTarget::OpenAiResponses,
    "OpenAI Responses"
);
define_request_bridge_wrappers!(
    feature = "openai",
    bridge_chat_request_to_openai_chat_completions_json,
    bridge_chat_request_to_openai_chat_completions_json_with_options,
    BridgeTarget::OpenAiChatCompletions,
    "OpenAI Chat Completions"
);
define_request_bridge_wrappers!(
    feature = "anthropic",
    bridge_chat_request_to_anthropic_messages_json,
    bridge_chat_request_to_anthropic_messages_json_with_options,
    BridgeTarget::AnthropicMessages,
    "Anthropic Messages"
);
define_request_bridge_wrappers!(
    any(feature = "google", feature = "google-vertex"),
    bridge_chat_request_to_gemini_generate_content_json,
    bridge_chat_request_to_gemini_generate_content_json_with_options,
    BridgeTarget::GeminiGenerateContent,
    "Gemini GenerateContent"
);

fn should_reject_request(
    options: &BridgeOptions,
    ctx: &siumai_core::bridge::RequestBridgeContext,
    report: &mut siumai_core::bridge::BridgeReport,
) -> bool {
    let action = options.loss_policy.request_action(ctx, report);
    reject_if_needed(report, action, "request", ctx.target)
}
