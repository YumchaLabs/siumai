//! Anthropic Claude Model Constants
//!
//! This module provides convenient constants for Anthropic Claude models, making it easy
//! for developers to reference specific models without hardcoding strings.

/// Claude Opus 4.1 model family constants (latest flagship)
pub mod claude_opus_4_1 {
    /// Claude Opus 4.1 - Most capable and intelligent model yet
    pub const CLAUDE_OPUS_4_1_20250805: &str = "claude-opus-4-1-20250805";
    /// Claude Opus 4.1 - Alias for latest
    pub const CLAUDE_OPUS_4_1: &str = "claude-opus-4-1";

    /// All Claude Opus 4.1 models
    pub const ALL: &[&str] = &[CLAUDE_OPUS_4_1_20250805, CLAUDE_OPUS_4_1];
}

/// Claude Opus 4 model family constants
pub mod claude_opus_4 {
    /// Claude Opus 4 - Previous flagship model
    pub const CLAUDE_OPUS_4_20250514: &str = "claude-opus-4-20250514";
    /// Claude Opus 4 - Alias
    pub const CLAUDE_OPUS_4_0: &str = "claude-opus-4-0";

    /// All Claude Opus 4 models
    pub const ALL: &[&str] = &[CLAUDE_OPUS_4_20250514, CLAUDE_OPUS_4_0];
}

/// Claude Sonnet 4 model family constants
pub mod claude_sonnet_4 {
    /// Claude Sonnet 4 - High-performance model with exceptional reasoning
    pub const CLAUDE_SONNET_4_20250514: &str = "claude-sonnet-4-20250514";
    /// Claude Sonnet 4 - Alias
    pub const CLAUDE_SONNET_4_0: &str = "claude-sonnet-4-0";

    /// All Claude Sonnet 4 models
    pub const ALL: &[&str] = &[CLAUDE_SONNET_4_20250514, CLAUDE_SONNET_4_0];
}

/// Claude Sonnet 3.7 model family constants
pub mod claude_sonnet_3_7 {
    /// Claude Sonnet 3.7 - High-performance model with early extended thinking
    pub const CLAUDE_3_7_SONNET_20250219: &str = "claude-3-7-sonnet-20250219";
    /// Claude Sonnet 3.7 - Latest alias
    pub const CLAUDE_3_7_SONNET_LATEST: &str = "claude-3-7-sonnet-latest";

    /// All Claude Sonnet 3.7 models
    pub const ALL: &[&str] = &[CLAUDE_3_7_SONNET_20250219, CLAUDE_3_7_SONNET_LATEST];
}

/// Claude Sonnet 3.5 model family constants
pub mod claude_sonnet_3_5 {
    /// Claude Sonnet 3.5 v2 - Latest version
    pub const CLAUDE_3_5_SONNET_20241022: &str = "claude-3-5-sonnet-20241022";
    /// Claude Sonnet 3.5 - Original version
    pub const CLAUDE_3_5_SONNET_20240620: &str = "claude-3-5-sonnet-20240620";
    /// Claude Sonnet 3.5 - Latest alias
    pub const CLAUDE_3_5_SONNET_LATEST: &str = "claude-3-5-sonnet-latest";

    /// All Claude Sonnet 3.5 models
    pub const ALL: &[&str] = &[
        CLAUDE_3_5_SONNET_20241022,
        CLAUDE_3_5_SONNET_20240620,
        CLAUDE_3_5_SONNET_LATEST,
    ];
}

/// Claude Haiku 3.5 model family constants
pub mod claude_haiku_3_5 {
    /// Claude Haiku 3.5 - Fastest model
    pub const CLAUDE_3_5_HAIKU_20241022: &str = "claude-3-5-haiku-20241022";
    /// Claude Haiku 3.5 - Latest alias
    pub const CLAUDE_3_5_HAIKU_LATEST: &str = "claude-3-5-haiku-latest";

    /// All Claude Haiku 3.5 models
    pub const ALL: &[&str] = &[CLAUDE_3_5_HAIKU_20241022, CLAUDE_3_5_HAIKU_LATEST];
}

/// Claude Haiku 3 model family constants
pub mod claude_haiku_3 {
    /// Claude Haiku 3 - Fast and compact model
    pub const CLAUDE_3_HAIKU_20240307: &str = "claude-3-haiku-20240307";

    /// All Claude Haiku 3 models
    pub const ALL: &[&str] = &[CLAUDE_3_HAIKU_20240307];
}

/// Claude Opus 3 model family constants (legacy)
pub mod claude_opus_3 {
    /// Claude Opus 3 - Legacy flagship model
    pub const CLAUDE_3_OPUS_20240229: &str = "claude-3-opus-20240229";

    /// All Claude Opus 3 models
    pub const ALL: &[&str] = &[CLAUDE_3_OPUS_20240229];
}

/// Claude Sonnet 3 model family constants (legacy)
pub mod claude_sonnet_3 {
    /// Claude Sonnet 3 - Legacy balanced model
    pub const CLAUDE_3_SONNET_20240229: &str = "claude-3-sonnet-20240229";

    /// All Claude Sonnet 3 models
    pub const ALL: &[&str] = &[CLAUDE_3_SONNET_20240229];
}

/// Popular model recommendations
pub mod popular {
    use super::*;

    /// Most capable model
    pub const FLAGSHIP: &str = claude_opus_4_1::CLAUDE_OPUS_4_1;
    /// Best balance of capability and performance
    pub const BALANCED: &str = claude_sonnet_4::CLAUDE_SONNET_4_20250514;
    /// Fastest model for quick responses
    pub const FAST: &str = claude_haiku_3_5::CLAUDE_3_5_HAIKU_LATEST;
    /// Best for thinking and reasoning
    pub const THINKING: &str = claude_sonnet_3_7::CLAUDE_3_7_SONNET_LATEST;
    /// Latest and most advanced
    pub const LATEST: &str = claude_opus_4_1::CLAUDE_OPUS_4_1;
}

/// Model capabilities by family
pub mod capabilities {
    /// Models with thinking capability
    pub const THINKING_MODELS: &[&str] = &[
        super::claude_opus_4_1::CLAUDE_OPUS_4_1_20250805,
        super::claude_opus_4::CLAUDE_OPUS_4_20250514,
        super::claude_sonnet_4::CLAUDE_SONNET_4_20250514,
        super::claude_sonnet_3_7::CLAUDE_3_7_SONNET_20250219,
    ];

    /// Models with vision capability
    pub const VISION_MODELS: &[&str] = &[
        super::claude_opus_4_1::CLAUDE_OPUS_4_1_20250805,
        super::claude_opus_4::CLAUDE_OPUS_4_20250514,
        super::claude_sonnet_4::CLAUDE_SONNET_4_20250514,
        super::claude_sonnet_3_7::CLAUDE_3_7_SONNET_20250219,
        super::claude_sonnet_3_5::CLAUDE_3_5_SONNET_20241022,
        super::claude_sonnet_3_5::CLAUDE_3_5_SONNET_20240620,
        super::claude_haiku_3_5::CLAUDE_3_5_HAIKU_20241022,
    ];

    /// Models with priority tier access
    pub const PRIORITY_TIER_MODELS: &[&str] = &[
        super::claude_opus_4_1::CLAUDE_OPUS_4_1_20250805,
        super::claude_opus_4::CLAUDE_OPUS_4_20250514,
        super::claude_sonnet_4::CLAUDE_SONNET_4_20250514,
        super::claude_sonnet_3_7::CLAUDE_3_7_SONNET_20250219,
        super::claude_sonnet_3_5::CLAUDE_3_5_SONNET_20241022,
        super::claude_haiku_3_5::CLAUDE_3_5_HAIKU_20241022,
    ];
}

/// Get all chat models
pub fn all_chat_models() -> Vec<&'static str> {
    let mut models = Vec::new();
    models.extend_from_slice(claude_opus_4_1::ALL);
    models.extend_from_slice(claude_opus_4::ALL);
    models.extend_from_slice(claude_sonnet_4::ALL);
    models.extend_from_slice(claude_sonnet_3_7::ALL);
    models.extend_from_slice(claude_sonnet_3_5::ALL);
    models.extend_from_slice(claude_haiku_3_5::ALL);
    models.extend_from_slice(claude_haiku_3::ALL);
    models.extend_from_slice(claude_opus_3::ALL);
    models.extend_from_slice(claude_sonnet_3::ALL);
    models
}

/// Get all models with thinking capability
pub fn all_thinking_models() -> Vec<&'static str> {
    capabilities::THINKING_MODELS.to_vec()
}

/// Get all models with vision capability
pub fn all_vision_models() -> Vec<&'static str> {
    capabilities::VISION_MODELS.to_vec()
}

/// Get all models with priority tier access
pub fn all_priority_tier_models() -> Vec<&'static str> {
    capabilities::PRIORITY_TIER_MODELS.to_vec()
}

/// Check if a model supports thinking
pub fn supports_thinking(model_id: &str) -> bool {
    capabilities::THINKING_MODELS.contains(&model_id)
}

/// Check if a model supports vision
pub fn supports_vision(model_id: &str) -> bool {
    capabilities::VISION_MODELS.contains(&model_id)
}

/// Check if a model has priority tier access
pub fn has_priority_tier(model_id: &str) -> bool {
    capabilities::PRIORITY_TIER_MODELS.contains(&model_id)
}

/// Get the context window size for a model
pub fn get_context_window(model_id: &str) -> u32 {
    match model_id {
        // Claude 4 models have 200K context
        id if id.contains("claude-opus-4") || id.contains("claude-sonnet-4") => 200_000,
        // Claude 3.7 has 200K context
        id if id.contains("claude-3-7-sonnet") => 200_000,
        // Claude 3.5 models have 200K context
        id if id.contains("claude-3-5") => 200_000,
        // Claude 3 models have 200K context
        id if id.contains("claude-3") => 200_000,
        // Default fallback
        _ => 200_000,
    }
}

/// Get the maximum output tokens for a model
pub fn get_max_output_tokens(model_id: &str) -> u32 {
    match model_id {
        // Claude 4 models have higher output limits
        id if id.contains("claude-opus-4") || id.contains("claude-sonnet-4") => 32_000,
        // Claude 3.7 has higher output limit
        id if id.contains("claude-3-7-sonnet") => 64_000,
        // Claude 3.5 models
        id if id.contains("claude-3-5") => 8192,
        // Claude 3 models
        id if id.contains("claude-3") => 4096,
        // Default fallback
        _ => 8192,
    }
}
