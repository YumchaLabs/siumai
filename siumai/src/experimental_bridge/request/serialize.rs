//! Request bridge serialization.

use crate::experimental_bridge::customize::apply_request_remapper;
use crate::experimental_bridge::planner::{RequestBridgePath, plan_chat_request_bridge};
use siumai_core::LlmError;
use siumai_core::bridge::{
    BridgeLossAction, BridgeMode, BridgeOptions, BridgeReport, BridgeResult, BridgeTarget,
    RequestBridgeContext,
};
use siumai_core::execution::transformers::request::RequestTransformer;
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
    let path_label = Some(match plan.path {
        RequestBridgePath::ViaNormalized => "via-normalized".to_string(),
        RequestBridgePath::Direct(pair) => pair.as_str().to_string(),
    });
    let ctx = RequestBridgeContext::new(
        source,
        target,
        options.mode,
        options.route_label.clone(),
        path_label,
    );
    let mut report = BridgeReport::with_source(source, target, options.mode);
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

/// Convenience wrapper for `OpenAI Responses`.
#[cfg(feature = "openai")]
pub fn bridge_chat_request_to_openai_responses_json(
    request: &ChatRequest,
    source: Option<BridgeTarget>,
    mode: BridgeMode,
) -> Result<BridgeResult<serde_json::Value>, LlmError> {
    bridge_chat_request_to_json(request, source, BridgeTarget::OpenAiResponses, mode)
}

/// Convenience wrapper for `OpenAI Responses` with bridge customization.
#[cfg(feature = "openai")]
pub fn bridge_chat_request_to_openai_responses_json_with_options(
    request: &ChatRequest,
    source: Option<BridgeTarget>,
    options: BridgeOptions,
) -> Result<BridgeResult<serde_json::Value>, LlmError> {
    bridge_chat_request_to_json_with_options(
        request,
        source,
        BridgeTarget::OpenAiResponses,
        options,
    )
}

/// Convenience wrapper for `OpenAI Chat Completions`.
#[cfg(feature = "openai")]
pub fn bridge_chat_request_to_openai_chat_completions_json(
    request: &ChatRequest,
    source: Option<BridgeTarget>,
    mode: BridgeMode,
) -> Result<BridgeResult<serde_json::Value>, LlmError> {
    bridge_chat_request_to_json(request, source, BridgeTarget::OpenAiChatCompletions, mode)
}

/// Convenience wrapper for `OpenAI Chat Completions` with bridge customization.
#[cfg(feature = "openai")]
pub fn bridge_chat_request_to_openai_chat_completions_json_with_options(
    request: &ChatRequest,
    source: Option<BridgeTarget>,
    options: BridgeOptions,
) -> Result<BridgeResult<serde_json::Value>, LlmError> {
    bridge_chat_request_to_json_with_options(
        request,
        source,
        BridgeTarget::OpenAiChatCompletions,
        options,
    )
}

/// Convenience wrapper for `Anthropic Messages`.
#[cfg(feature = "anthropic")]
pub fn bridge_chat_request_to_anthropic_messages_json(
    request: &ChatRequest,
    source: Option<BridgeTarget>,
    mode: BridgeMode,
) -> Result<BridgeResult<serde_json::Value>, LlmError> {
    bridge_chat_request_to_json(request, source, BridgeTarget::AnthropicMessages, mode)
}

/// Convenience wrapper for `Anthropic Messages` with bridge customization.
#[cfg(feature = "anthropic")]
pub fn bridge_chat_request_to_anthropic_messages_json_with_options(
    request: &ChatRequest,
    source: Option<BridgeTarget>,
    options: BridgeOptions,
) -> Result<BridgeResult<serde_json::Value>, LlmError> {
    bridge_chat_request_to_json_with_options(
        request,
        source,
        BridgeTarget::AnthropicMessages,
        options,
    )
}

/// Convenience wrapper for `Gemini GenerateContent`.
#[cfg(feature = "google")]
pub fn bridge_chat_request_to_gemini_generate_content_json(
    request: &ChatRequest,
    source: Option<BridgeTarget>,
    mode: BridgeMode,
) -> Result<BridgeResult<serde_json::Value>, LlmError> {
    bridge_chat_request_to_json(request, source, BridgeTarget::GeminiGenerateContent, mode)
}

/// Convenience wrapper for `Gemini GenerateContent` with bridge customization.
#[cfg(feature = "google")]
pub fn bridge_chat_request_to_gemini_generate_content_json_with_options(
    request: &ChatRequest,
    source: Option<BridgeTarget>,
    options: BridgeOptions,
) -> Result<BridgeResult<serde_json::Value>, LlmError> {
    bridge_chat_request_to_json_with_options(
        request,
        source,
        BridgeTarget::GeminiGenerateContent,
        options,
    )
}

fn should_reject_request(
    options: &BridgeOptions,
    ctx: &RequestBridgeContext,
    report: &mut BridgeReport,
) -> bool {
    if report.is_rejected() {
        return true;
    }

    if matches!(
        options.loss_policy.request_action(ctx, report),
        BridgeLossAction::Reject
    ) {
        report.reject(format!(
            "bridge policy rejected request conversion to {}",
            ctx.target.as_str()
        ));
        return true;
    }

    false
}

fn transform_chat_request_to_json(
    request: &ChatRequest,
    target: BridgeTarget,
) -> Result<serde_json::Value, LlmError> {
    match target {
        BridgeTarget::OpenAiResponses => {
            #[cfg(feature = "openai")]
            {
                let tx =
                    siumai_protocol_openai::standards::openai::transformers::request::OpenAiResponsesRequestTransformer;
                tx.transform_chat(request)
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
                let tx =
                    siumai_protocol_openai::standards::openai::transformers::request::OpenAiRequestTransformer;
                tx.transform_chat(request)
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
                let tx =
                    siumai_protocol_anthropic::standards::anthropic::transformers::AnthropicRequestTransformer::default();
                tx.transform_chat(request)
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
                let tx =
                    siumai_protocol_gemini::standards::gemini::transformers::GeminiRequestTransformer {
                        config:
                            siumai_protocol_gemini::standards::gemini::types::GeminiConfig::default(),
                    };
                tx.transform_chat(request)
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
