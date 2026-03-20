//! Request bridge serialization.

use crate::experimental_bridge::planner::{RequestBridgePath, plan_chat_request_bridge};
use siumai_core::LlmError;
use siumai_core::bridge::{BridgeMode, BridgeReport, BridgeResult, BridgeTarget};
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
    let plan = plan_chat_request_bridge(source, target);
    let mut report = BridgeReport::with_source(source, target, mode);
    inspect_chat_request_bridge(request, target, &mut report);

    if matches!(mode, BridgeMode::Strict) && report.is_lossy() {
        report.reject(format!(
            "strict bridge mode rejected lossy request conversion to {}",
            target.as_str()
        ));
        return Ok(BridgeResult::rejected(report));
    }

    let value = match plan.path {
        RequestBridgePath::ViaNormalized => transform_chat_request_to_json(request, target)?,
        RequestBridgePath::Direct(pair) => {
            serialize_direct_request_bridge_pair(pair, request, &mut report)?
        }
    };

    if matches!(mode, BridgeMode::Strict) && report.is_lossy() {
        report.reject(format!(
            "strict bridge mode rejected lossy request conversion to {}",
            target.as_str()
        ));
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

/// Convenience wrapper for `OpenAI Chat Completions`.
#[cfg(feature = "openai")]
pub fn bridge_chat_request_to_openai_chat_completions_json(
    request: &ChatRequest,
    source: Option<BridgeTarget>,
    mode: BridgeMode,
) -> Result<BridgeResult<serde_json::Value>, LlmError> {
    bridge_chat_request_to_json(request, source, BridgeTarget::OpenAiChatCompletions, mode)
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

/// Convenience wrapper for `Gemini GenerateContent`.
#[cfg(feature = "google")]
pub fn bridge_chat_request_to_gemini_generate_content_json(
    request: &ChatRequest,
    source: Option<BridgeTarget>,
    mode: BridgeMode,
) -> Result<BridgeResult<serde_json::Value>, LlmError> {
    bridge_chat_request_to_json(request, source, BridgeTarget::GeminiGenerateContent, mode)
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
