pub use siumai_provider_anthropic::providers::anthropic::{
    AnthropicBuilder, AnthropicClient, AnthropicConfig, AnthropicProviderSettings, VERSION,
};

/// Create the Anthropic provider builder.
pub fn anthropic() -> AnthropicBuilder {
    crate::Provider::anthropic()
}

/// Create the Anthropic provider builder.
///
/// This is the Rust package-surface analogue of AI SDK `createAnthropic()`.
pub fn create_anthropic() -> AnthropicBuilder {
    anthropic()
}

/// Provider tool factories that return `Tool` directly (Vercel-aligned).
pub mod tools {
    pub use crate::tools::anthropic::*;
}

/// Provider-executed tool builders (typed args).
pub mod hosted_tools {
    pub use crate::hosted_tools::anthropic::*;
}

/// Compatibility alias for older imports.
pub mod provider_tools {
    pub use crate::tools::anthropic::*;
}

/// Typed provider options (`provider_options_map["anthropic"]`).
pub mod options {
    #[allow(deprecated)]
    pub use siumai_provider_anthropic::provider_options::anthropic::{
        AnthropicCacheControl, AnthropicCacheType, AnthropicContainerConfig,
        AnthropicContainerSkill, AnthropicContainerSkillType, AnthropicContextManagementAllKeep,
        AnthropicContextManagementConfig, AnthropicContextManagementEdit,
        AnthropicContextManagementInputTokensValue, AnthropicContextManagementThinkingKeep,
        AnthropicContextManagementThinkingTurnsKeep,
        AnthropicContextManagementThinkingTurnsKeepKind, AnthropicContextManagementToolUsesKeep,
        AnthropicContextManagementTrigger, AnthropicEffort, AnthropicInferenceGeo,
        AnthropicLanguageModelOptions, AnthropicMcpServer, AnthropicMcpServerType,
        AnthropicMcpToolConfiguration, AnthropicOptions, AnthropicProviderOptions,
        AnthropicRequestCacheControl, AnthropicRequestCacheControlTtl,
        AnthropicRequestCacheControlType, AnthropicRequestMetadata, AnthropicResponseFormat,
        AnthropicSpeed, AnthropicStructuredOutputMode, AnthropicTaskBudget,
        AnthropicTaskBudgetType, AnthropicThinkingConfig, AnthropicThinkingDisplay,
        AnthropicToolAllowedCaller, AnthropicToolOptions, PromptCachingConfig, ThinkingModeConfig,
    };
    pub use siumai_provider_anthropic::providers::anthropic::ext::AnthropicChatRequestExt;
}
#[allow(deprecated)]
pub use options::{
    AnthropicCacheControl, AnthropicCacheType, AnthropicChatRequestExt, AnthropicContainerConfig,
    AnthropicContainerSkill, AnthropicContainerSkillType, AnthropicContextManagementAllKeep,
    AnthropicContextManagementConfig, AnthropicContextManagementEdit,
    AnthropicContextManagementInputTokensValue, AnthropicContextManagementThinkingKeep,
    AnthropicContextManagementThinkingTurnsKeep, AnthropicContextManagementThinkingTurnsKeepKind,
    AnthropicContextManagementToolUsesKeep, AnthropicContextManagementTrigger, AnthropicEffort,
    AnthropicInferenceGeo, AnthropicLanguageModelOptions, AnthropicMcpServer,
    AnthropicMcpServerType, AnthropicMcpToolConfiguration, AnthropicOptions,
    AnthropicProviderOptions, AnthropicRequestCacheControl, AnthropicRequestCacheControlTtl,
    AnthropicRequestCacheControlType, AnthropicRequestMetadata, AnthropicResponseFormat,
    AnthropicSpeed, AnthropicStructuredOutputMode, AnthropicTaskBudget, AnthropicTaskBudgetType,
    AnthropicThinkingConfig, AnthropicThinkingDisplay, AnthropicToolAllowedCaller,
    AnthropicToolOptions, PromptCachingConfig, ThinkingModeConfig,
};

/// Typed response metadata helpers (`ChatResponse.provider_metadata["anthropic"]`).
pub mod metadata {
    pub use siumai_provider_anthropic::provider_metadata::anthropic::{
        AnthropicChatResponseExt, AnthropicCitation, AnthropicCitationsBlock,
        AnthropicContentPartExt, AnthropicMessageContainerMetadata, AnthropicMessageContainerSkill,
        AnthropicMessageMetadata, AnthropicMetadata, AnthropicServerToolUse, AnthropicSource,
        AnthropicToolCallMetadata, AnthropicToolCaller, AnthropicUsageIteration,
    };
}
pub use metadata::{
    AnthropicChatResponseExt, AnthropicCitation, AnthropicCitationsBlock, AnthropicContentPartExt,
    AnthropicMessageContainerMetadata, AnthropicMessageContainerSkill, AnthropicMessageMetadata,
    AnthropicMetadata, AnthropicServerToolUse, AnthropicSource, AnthropicToolCallMetadata,
    AnthropicToolCaller, AnthropicUsageIteration,
};
pub use siumai_provider_anthropic::providers::anthropic::{
    find_anthropic_container_id_from_last_step, forward_anthropic_container_id_from_last_step,
};

/// Non-unified Anthropic extension APIs (request extensions, tool helpers, thinking, etc.).
pub mod ext {
    pub use siumai_provider_anthropic::providers::anthropic::ext::{
        AnthropicToolExt, structured_output, thinking, tools,
    };
}
pub use ext::AnthropicToolExt;

/// Provider-specific resources not covered by the unified families.
pub mod resources {
    pub use siumai_provider_anthropic::providers::anthropic::{
        AnthropicCountTokensResponse, AnthropicCreateMessageBatchRequest, AnthropicFiles,
        AnthropicListMessageBatchesResponse, AnthropicMessageBatch, AnthropicMessageBatchRequest,
        AnthropicMessageBatches, AnthropicSkills, AnthropicTokens,
    };
}

// Legacy Anthropic parameter structs (provider-owned).
pub use siumai_provider_anthropic::params::anthropic::{AnthropicParams, CacheControl};
