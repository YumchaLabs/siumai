//! Anthropic Provider Module
//!
//! Modular implementation of Anthropic Claude API client with capability separation.

pub mod builder;
pub mod cache;
pub mod client;
pub mod config;
/// Anthropic extension APIs (non-unified surface)
pub mod ext;
pub mod files;
pub mod message_batches;
pub mod middleware;
pub mod model_constants;
pub mod models;
pub mod prepare_step;
pub mod settings;
pub mod skills;
pub mod spec;
pub mod streaming;
pub mod thinking;
pub mod tokens;
pub mod transformers;
pub mod types;
pub mod utils;

// Re-export main types for backward compatibility
pub use builder::AnthropicBuilder;
pub use client::AnthropicClient;
pub use config::*;
pub use files::AnthropicFiles;
pub use message_batches::{
    AnthropicCreateMessageBatchRequest, AnthropicListMessageBatchesResponse, AnthropicMessageBatch,
    AnthropicMessageBatchRequest, AnthropicMessageBatches,
};
pub use middleware::AnthropicToolWarningsMiddleware;
pub use prepare_step::{
    find_anthropic_container_id_from_last_step, forward_anthropic_container_id_from_last_step,
};
pub use skills::AnthropicSkills;
pub use tokens::{AnthropicCountTokensResponse, AnthropicTokens};
pub use types::*;

// Provider-owned typed options (kept out of `siumai-core`).
pub use crate::provider_options::anthropic::{
    AnthropicCacheControl, AnthropicCacheType, AnthropicContainerConfig, AnthropicContainerSkill,
    AnthropicContainerSkillType, AnthropicContextManagementAllKeep,
    AnthropicContextManagementConfig, AnthropicContextManagementEdit,
    AnthropicContextManagementInputTokensValue, AnthropicContextManagementThinkingKeep,
    AnthropicContextManagementThinkingTurnsKeep, AnthropicContextManagementThinkingTurnsKeepKind,
    AnthropicContextManagementToolUsesKeep, AnthropicContextManagementTrigger, AnthropicEffort,
    AnthropicInferenceGeo, AnthropicMcpServer, AnthropicMcpServerType,
    AnthropicMcpToolConfiguration, AnthropicOptions, AnthropicRequestCacheControl,
    AnthropicRequestCacheControlTtl, AnthropicRequestCacheControlType, AnthropicRequestMetadata,
    AnthropicResponseFormat, AnthropicSpeed, AnthropicStructuredOutputMode, AnthropicTaskBudget,
    AnthropicTaskBudgetType, AnthropicThinkingConfig, AnthropicThinkingDisplay,
    PromptCachingConfig, ThinkingModeConfig,
};

// Typed provider metadata views (protocol-owned; re-exported via this provider for ergonomics).
pub use crate::provider_metadata::anthropic::{
    AnthropicChatResponseExt, AnthropicCitation, AnthropicCitationsBlock, AnthropicContentPartExt,
    AnthropicMessageContainerMetadata, AnthropicMessageContainerSkill, AnthropicMessageMetadata,
    AnthropicMetadata, AnthropicServerToolUse, AnthropicSource, AnthropicToolCallMetadata,
    AnthropicToolCaller,
};

// Re-export capability implementations
pub use models::AnthropicModels;
pub use settings::AnthropicProviderSettings;

/// Rust package version exposed on the Anthropic package-surface facade.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub(crate) fn specific_params_from_legacy_params(
    params: &crate::params::AnthropicParams,
) -> crate::providers::anthropic::types::AnthropicSpecificParams {
    crate::providers::anthropic::types::AnthropicSpecificParams {
        beta_features: params.beta_features.clone().unwrap_or_default(),
        // Legacy params are retained for backward compatibility; prompt caching is modeled as a
        // modern request-level provider option (`providerOptions["anthropic"]`) for new code.
        cache_control: params
            .cache_control
            .as_ref()
            .map(|_cc| crate::providers::anthropic::cache::CacheControl::ephemeral()),
        thinking_config: params
            .thinking_budget
            .map(crate::providers::anthropic::thinking::ThinkingConfig::enabled),
        metadata: params.metadata.as_ref().map(|m| {
            let mut json_map = serde_json::Map::new();
            for (k, v) in m {
                json_map.insert(k.clone(), serde_json::Value::String(v.clone()));
            }
            serde_json::Value::Object(json_map)
        }),
    }
}
