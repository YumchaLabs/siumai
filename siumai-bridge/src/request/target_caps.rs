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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RequestToolApprovalResponseMode {
    PreserveProviderExecutedOnly,
    DropAll,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct RequestTargetCapabilities {
    pub reasoning_mode: RequestReasoningMode,
    pub cache_control_mode: RequestCacheControlMode,
    pub tool_approval_response_mode: RequestToolApprovalResponseMode,
}

pub(super) const fn request_target_capabilities(target: BridgeTarget) -> RequestTargetCapabilities {
    match target {
        BridgeTarget::OpenAiResponses => RequestTargetCapabilities {
            reasoning_mode: RequestReasoningMode::OpenAiResponses,
            cache_control_mode: RequestCacheControlMode::DropAnthropicControls,
            tool_approval_response_mode:
                RequestToolApprovalResponseMode::PreserveProviderExecutedOnly,
        },
        BridgeTarget::OpenAiChatCompletions => RequestTargetCapabilities {
            reasoning_mode: RequestReasoningMode::OpenAiChatCompletions,
            cache_control_mode: RequestCacheControlMode::DropAnthropicControls,
            tool_approval_response_mode: RequestToolApprovalResponseMode::DropAll,
        },
        BridgeTarget::AnthropicMessages => RequestTargetCapabilities {
            reasoning_mode: RequestReasoningMode::AnthropicMessages,
            cache_control_mode: RequestCacheControlMode::AnthropicLimit4,
            tool_approval_response_mode: RequestToolApprovalResponseMode::DropAll,
        },
        BridgeTarget::GeminiGenerateContent => RequestTargetCapabilities {
            reasoning_mode: RequestReasoningMode::Preserve,
            cache_control_mode: RequestCacheControlMode::DropAnthropicControls,
            tool_approval_response_mode: RequestToolApprovalResponseMode::DropAll,
        },
    }
}
