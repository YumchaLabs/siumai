//! Shared response-target capability definitions for bridge inspection.

use siumai_core::bridge::BridgeTarget;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ResponseUsageMode {
    PreserveAll,
    AnthropicAggregateOnly,
    GeminiPartialBreakdown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ResponseFinishReasonMode {
    OpenAiFamily,
    AnthropicMessages,
    GeminiGenerateContent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ResponseContentPartProviderMetadataMode {
    OpenAiResponses,
    AnthropicMessages,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ResponseProviderMetadataMode {
    OpenAiResponses,
    AnthropicMessages,
    None,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ResponseTargetCapabilities {
    pub target: BridgeTarget,
    pub supports_reasoning_blocks: bool,
    pub supports_system_fingerprint: bool,
    pub supports_service_tier: bool,
    pub supports_provider_executed_tool_results: bool,
    pub usage_mode: ResponseUsageMode,
    pub finish_reason_mode: ResponseFinishReasonMode,
    pub content_part_provider_metadata_mode: ResponseContentPartProviderMetadataMode,
    pub provider_metadata_mode: ResponseProviderMetadataMode,
}

pub(super) const fn response_target_capabilities(
    target: BridgeTarget,
) -> ResponseTargetCapabilities {
    match target {
        BridgeTarget::OpenAiResponses => ResponseTargetCapabilities {
            target,
            supports_reasoning_blocks: true,
            supports_system_fingerprint: true,
            supports_service_tier: true,
            supports_provider_executed_tool_results: true,
            usage_mode: ResponseUsageMode::PreserveAll,
            finish_reason_mode: ResponseFinishReasonMode::OpenAiFamily,
            content_part_provider_metadata_mode:
                ResponseContentPartProviderMetadataMode::OpenAiResponses,
            provider_metadata_mode: ResponseProviderMetadataMode::OpenAiResponses,
        },
        BridgeTarget::OpenAiChatCompletions => ResponseTargetCapabilities {
            target,
            supports_reasoning_blocks: false,
            supports_system_fingerprint: true,
            supports_service_tier: true,
            supports_provider_executed_tool_results: false,
            usage_mode: ResponseUsageMode::PreserveAll,
            finish_reason_mode: ResponseFinishReasonMode::OpenAiFamily,
            content_part_provider_metadata_mode: ResponseContentPartProviderMetadataMode::None,
            provider_metadata_mode: ResponseProviderMetadataMode::None,
        },
        BridgeTarget::AnthropicMessages => ResponseTargetCapabilities {
            target,
            supports_reasoning_blocks: true,
            supports_system_fingerprint: false,
            supports_service_tier: true,
            supports_provider_executed_tool_results: false,
            usage_mode: ResponseUsageMode::AnthropicAggregateOnly,
            finish_reason_mode: ResponseFinishReasonMode::AnthropicMessages,
            content_part_provider_metadata_mode:
                ResponseContentPartProviderMetadataMode::AnthropicMessages,
            provider_metadata_mode: ResponseProviderMetadataMode::AnthropicMessages,
        },
        BridgeTarget::GeminiGenerateContent => ResponseTargetCapabilities {
            target,
            supports_reasoning_blocks: false,
            supports_system_fingerprint: false,
            supports_service_tier: false,
            supports_provider_executed_tool_results: false,
            usage_mode: ResponseUsageMode::GeminiPartialBreakdown,
            finish_reason_mode: ResponseFinishReasonMode::GeminiGenerateContent,
            content_part_provider_metadata_mode: ResponseContentPartProviderMetadataMode::None,
            provider_metadata_mode: ResponseProviderMetadataMode::None,
        },
    }
}
