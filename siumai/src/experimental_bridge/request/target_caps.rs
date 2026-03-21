//! Shared request-target capability definitions for bridge inspection.

use siumai_core::bridge::BridgeTarget;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RequestReasoningMode {
    OpenAiResponses,
    OpenAiChatCompletions,
    AnthropicMessages,
    Preserve,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RequestCacheControlMode {
    AnthropicLimit4,
    DropAnthropicControls,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct RequestTargetCapabilities {
    pub reasoning_mode: RequestReasoningMode,
    pub cache_control_mode: RequestCacheControlMode,
    pub preserves_tool_approval_responses: bool,
}

pub(super) const fn request_target_capabilities(target: BridgeTarget) -> RequestTargetCapabilities {
    match target {
        BridgeTarget::OpenAiResponses => RequestTargetCapabilities {
            reasoning_mode: RequestReasoningMode::OpenAiResponses,
            cache_control_mode: RequestCacheControlMode::DropAnthropicControls,
            preserves_tool_approval_responses: true,
        },
        BridgeTarget::OpenAiChatCompletions => RequestTargetCapabilities {
            reasoning_mode: RequestReasoningMode::OpenAiChatCompletions,
            cache_control_mode: RequestCacheControlMode::DropAnthropicControls,
            preserves_tool_approval_responses: false,
        },
        BridgeTarget::AnthropicMessages => RequestTargetCapabilities {
            reasoning_mode: RequestReasoningMode::AnthropicMessages,
            cache_control_mode: RequestCacheControlMode::AnthropicLimit4,
            preserves_tool_approval_responses: false,
        },
        BridgeTarget::GeminiGenerateContent => RequestTargetCapabilities {
            reasoning_mode: RequestReasoningMode::Preserve,
            cache_control_mode: RequestCacheControlMode::DropAnthropicControls,
            preserves_tool_approval_responses: false,
        },
    }
}
