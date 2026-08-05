//! Anthropic Claude Model Constants
//!
//! This module provides convenient constants for Anthropic Claude models, making it easy
//! for developers to reference specific models without hardcoding strings.

/// Primary official source for the curated model catalog.
pub const MODEL_CATALOG_SOURCE: &str =
    "https://platform.claude.com/docs/en/about-claude/models/overview";
/// Date on which the current-model policies were verified.
pub const MODEL_CATALOG_VERIFIED_ON: &str = "2026-08-05";

/// Whether an audited model profile makes a capability claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilitySupport {
    /// The official model documentation explicitly supports the capability.
    Supported,
    /// The official model documentation explicitly rejects the capability.
    Unsupported,
    /// The selected official sources do not establish a stable answer.
    Unknown,
}

/// Thinking behavior exposed by a model family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingSupport {
    /// The model does not expose Anthropic thinking controls.
    Unsupported,
    /// Thinking is enabled by the model and can only be configured through adaptive thinking.
    AdaptiveOnly,
    /// The model accepts the legacy explicit thinking budget controls.
    Manual,
    /// The model accepts both adaptive and explicit thinking controls.
    AdaptiveAndManual,
}

/// Whether adaptive thinking may be disabled for a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingDisablePolicy {
    /// `thinking.type=disabled` is accepted.
    Allowed,
    /// Thinking is always enabled and cannot be disabled.
    Forbidden,
    /// Disabling thinking is accepted only through `high` effort.
    AllowedThroughHighEffort,
    /// The curated sources do not establish a stable rule.
    Unknown,
}

/// Whether the standard sampling controls are accepted by a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingSupport {
    /// Sampling controls are accepted.
    Supported,
    /// Anthropic rejects sampling controls for this model family.
    Unsupported,
}

/// A conservative, provider-owned model profile.
///
/// Profiles are intentionally keyed by exact model IDs (including the aliases exported below).
/// An ID that is not in the catalog is still callable, but has no inferred limits or optional
/// capabilities. This keeps model listing useful without turning an unknown future model into a
/// false Claude 3/4 capability claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnthropicModelProfile {
    /// Maximum input context in tokens.
    pub context_window: u32,
    /// Maximum generated output in tokens.
    pub max_output_tokens: u32,
    /// Thinking support and its request policy.
    pub thinking: ThinkingSupport,
    /// Whether and under which conditions adaptive thinking may be disabled.
    pub thinking_disable: ThinkingDisablePolicy,
    /// Whether image/document vision inputs are supported.
    pub supports_vision: bool,
    /// Whether tool/function calling is supported.
    pub supports_tools: bool,
    /// Whether server-sent streaming is supported.
    pub supports_streaming: bool,
    /// Whether prompt cache control can be attached to request content.
    pub supports_prompt_caching: bool,
    /// Whether assistant prefill is supported.
    pub supports_prefill: CapabilitySupport,
    /// Whether the Anthropic priority tier is documented for this model.
    pub supports_priority_tier: CapabilitySupport,
    /// Whether temperature/top-p/top-k controls are accepted.
    pub sampling: SamplingSupport,
}

const fn current_model_profile(
    thinking: ThinkingSupport,
    thinking_disable: ThinkingDisablePolicy,
    supports_prefill: CapabilitySupport,
    supports_priority_tier: CapabilitySupport,
) -> AnthropicModelProfile {
    AnthropicModelProfile {
        context_window: 1_000_000,
        max_output_tokens: 128_000,
        thinking,
        thinking_disable,
        supports_vision: true,
        supports_tools: true,
        supports_streaming: true,
        supports_prompt_caching: true,
        supports_prefill,
        supports_priority_tier,
        sampling: SamplingSupport::Unsupported,
    }
}

const fn legacy_model_profile(
    context_window: u32,
    max_output_tokens: u32,
    thinking: ThinkingSupport,
    supports_vision: bool,
    supports_priority_tier: bool,
) -> AnthropicModelProfile {
    AnthropicModelProfile {
        context_window,
        max_output_tokens,
        thinking,
        thinking_disable: ThinkingDisablePolicy::Allowed,
        supports_vision,
        supports_tools: true,
        supports_streaming: true,
        supports_prompt_caching: true,
        supports_prefill: CapabilitySupport::Supported,
        supports_priority_tier: if supports_priority_tier {
            CapabilitySupport::Supported
        } else {
            CapabilitySupport::Unsupported
        },
        sampling: SamplingSupport::Supported,
    }
}

/// Claude Opus 5 model family constants (current flagship).
pub mod claude_opus_5 {
    /// Claude Opus 5.
    pub const CLAUDE_OPUS_5: &str = "claude-opus-5";

    /// All Claude Opus 5 models.
    pub const ALL: &[&str] = &[CLAUDE_OPUS_5];
}

/// Claude Sonnet 5 model family constants (current balanced model).
pub mod claude_sonnet_5 {
    /// Claude Sonnet 5.
    pub const CLAUDE_SONNET_5: &str = "claude-sonnet-5";

    /// All Claude Sonnet 5 models.
    pub const ALL: &[&str] = &[CLAUDE_SONNET_5];
}

/// Claude Fable 5 model family constants (current always-thinking model).
pub mod claude_fable_5 {
    /// Claude Fable 5.
    pub const CLAUDE_FABLE_5: &str = "claude-fable-5";

    /// All Claude Fable 5 models.
    pub const ALL: &[&str] = &[CLAUDE_FABLE_5];
}

/// Claude Mythos 5 model family constants (limited-availability model).
pub mod claude_mythos_5 {
    /// Claude Mythos 5.
    pub const CLAUDE_MYTHOS_5: &str = "claude-mythos-5";

    /// All Claude Mythos 5 models.
    pub const ALL: &[&str] = &[CLAUDE_MYTHOS_5];
}

/// Claude Opus 4.7 model family constants (latest flagship)
pub mod claude_opus_4_7 {
    /// Claude Opus 4.7 - Alias for latest
    pub const CLAUDE_OPUS_4_7: &str = "claude-opus-4-7";

    /// All Claude Opus 4.7 models
    pub const ALL: &[&str] = &[CLAUDE_OPUS_4_7];
}

/// Claude Opus 4.6 model family constants
pub mod claude_opus_4_6 {
    /// Claude Opus 4.6 - Alias for latest
    pub const CLAUDE_OPUS_4_6: &str = "claude-opus-4-6";

    /// All Claude Opus 4.6 models
    pub const ALL: &[&str] = &[CLAUDE_OPUS_4_6];
}

/// Claude Opus 4.5 model family constants
pub mod claude_opus_4_5 {
    /// Claude Opus 4.5 - Specific version
    pub const CLAUDE_OPUS_4_5_20251101: &str = "claude-opus-4-5-20251101";
    /// Claude Opus 4.5 - Alias
    pub const CLAUDE_OPUS_4_5: &str = "claude-opus-4-5";

    /// All Claude Opus 4.5 models
    pub const ALL: &[&str] = &[CLAUDE_OPUS_4_5_20251101, CLAUDE_OPUS_4_5];
}

/// Claude Opus 4.1 model family constants
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

/// Claude Sonnet 4.6 model family constants
pub mod claude_sonnet_4_6 {
    /// Claude Sonnet 4.6 - Alias for latest
    pub const CLAUDE_SONNET_4_6: &str = "claude-sonnet-4-6";

    /// All Claude Sonnet 4.6 models
    pub const ALL: &[&str] = &[CLAUDE_SONNET_4_6];
}

/// Claude Sonnet 4.5 model family constants
pub mod claude_sonnet_4_5 {
    /// Claude Sonnet 4.5 - Specific version
    pub const CLAUDE_SONNET_4_5_20250929: &str = "claude-sonnet-4-5-20250929";
    /// Claude Sonnet 4.5 - Alias
    pub const CLAUDE_SONNET_4_5: &str = "claude-sonnet-4-5";

    /// All Claude Sonnet 4.5 models
    pub const ALL: &[&str] = &[CLAUDE_SONNET_4_5_20250929, CLAUDE_SONNET_4_5];
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

/// Claude Haiku 4.5 model family constants
pub mod claude_haiku_4_5 {
    /// Claude Haiku 4.5 - Specific version
    pub const CLAUDE_HAIKU_4_5_20251001: &str = "claude-haiku-4-5-20251001";
    /// Claude Haiku 4.5 - Alias
    pub const CLAUDE_HAIKU_4_5: &str = "claude-haiku-4-5";

    /// All Claude Haiku 4.5 models
    pub const ALL: &[&str] = &[CLAUDE_HAIKU_4_5_20251001, CLAUDE_HAIKU_4_5];
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
    pub const FLAGSHIP: &str = claude_fable_5::CLAUDE_FABLE_5;
    /// Best balance of capability and performance
    pub const BALANCED: &str = claude_sonnet_5::CLAUDE_SONNET_5;
    /// Fastest model for quick responses
    pub const FAST: &str = claude_haiku_4_5::CLAUDE_HAIKU_4_5;
    /// Best for thinking and reasoning
    pub const THINKING: &str = claude_fable_5::CLAUDE_FABLE_5;
    /// Latest and most advanced
    pub const LATEST: &str = claude_fable_5::CLAUDE_FABLE_5;
}

/// Model capabilities by family
pub mod capabilities {
    /// Models with thinking capability
    pub const THINKING_MODELS: &[&str] = &[
        super::claude_opus_5::CLAUDE_OPUS_5,
        super::claude_sonnet_5::CLAUDE_SONNET_5,
        super::claude_fable_5::CLAUDE_FABLE_5,
        super::claude_mythos_5::CLAUDE_MYTHOS_5,
        super::claude_opus_4_7::CLAUDE_OPUS_4_7,
        super::claude_opus_4_6::CLAUDE_OPUS_4_6,
        super::claude_opus_4_5::CLAUDE_OPUS_4_5_20251101,
        super::claude_opus_4_5::CLAUDE_OPUS_4_5,
        super::claude_opus_4_1::CLAUDE_OPUS_4_1_20250805,
        super::claude_sonnet_4_6::CLAUDE_SONNET_4_6,
        super::claude_sonnet_4_5::CLAUDE_SONNET_4_5_20250929,
        super::claude_sonnet_4_5::CLAUDE_SONNET_4_5,
        super::claude_opus_4::CLAUDE_OPUS_4_20250514,
        super::claude_sonnet_4::CLAUDE_SONNET_4_20250514,
        super::claude_haiku_4_5::CLAUDE_HAIKU_4_5_20251001,
        super::claude_haiku_4_5::CLAUDE_HAIKU_4_5,
        super::claude_sonnet_3_7::CLAUDE_3_7_SONNET_20250219,
    ];

    /// Models with vision capability
    pub const VISION_MODELS: &[&str] = &[
        super::claude_opus_5::CLAUDE_OPUS_5,
        super::claude_sonnet_5::CLAUDE_SONNET_5,
        super::claude_fable_5::CLAUDE_FABLE_5,
        super::claude_mythos_5::CLAUDE_MYTHOS_5,
        super::claude_opus_4_7::CLAUDE_OPUS_4_7,
        super::claude_opus_4_6::CLAUDE_OPUS_4_6,
        super::claude_opus_4_5::CLAUDE_OPUS_4_5_20251101,
        super::claude_opus_4_5::CLAUDE_OPUS_4_5,
        super::claude_opus_4_1::CLAUDE_OPUS_4_1_20250805,
        super::claude_sonnet_4_6::CLAUDE_SONNET_4_6,
        super::claude_sonnet_4_5::CLAUDE_SONNET_4_5_20250929,
        super::claude_sonnet_4_5::CLAUDE_SONNET_4_5,
        super::claude_opus_4::CLAUDE_OPUS_4_20250514,
        super::claude_sonnet_4::CLAUDE_SONNET_4_20250514,
        super::claude_haiku_4_5::CLAUDE_HAIKU_4_5_20251001,
        super::claude_haiku_4_5::CLAUDE_HAIKU_4_5,
        super::claude_sonnet_3_7::CLAUDE_3_7_SONNET_20250219,
        super::claude_sonnet_3_5::CLAUDE_3_5_SONNET_20241022,
        super::claude_sonnet_3_5::CLAUDE_3_5_SONNET_20240620,
        super::claude_haiku_3_5::CLAUDE_3_5_HAIKU_20241022,
    ];

    /// Models with priority tier access
    pub const PRIORITY_TIER_MODELS: &[&str] = &[
        super::claude_opus_4_7::CLAUDE_OPUS_4_7,
        super::claude_opus_4_6::CLAUDE_OPUS_4_6,
        super::claude_opus_4_5::CLAUDE_OPUS_4_5_20251101,
        super::claude_opus_4_5::CLAUDE_OPUS_4_5,
        super::claude_opus_4_1::CLAUDE_OPUS_4_1_20250805,
        super::claude_sonnet_4_6::CLAUDE_SONNET_4_6,
        super::claude_sonnet_4_5::CLAUDE_SONNET_4_5_20250929,
        super::claude_sonnet_4_5::CLAUDE_SONNET_4_5,
        super::claude_opus_4::CLAUDE_OPUS_4_20250514,
        super::claude_sonnet_4::CLAUDE_SONNET_4_20250514,
        super::claude_haiku_4_5::CLAUDE_HAIKU_4_5_20251001,
        super::claude_haiku_4_5::CLAUDE_HAIKU_4_5,
        super::claude_sonnet_3_7::CLAUDE_3_7_SONNET_20250219,
        super::claude_sonnet_3_5::CLAUDE_3_5_SONNET_20241022,
        super::claude_haiku_3_5::CLAUDE_3_5_HAIKU_20241022,
    ];
}

/// Get all chat models
pub fn all_chat_models() -> Vec<&'static str> {
    let mut models = Vec::new();
    models.extend_from_slice(claude_opus_5::ALL);
    models.extend_from_slice(claude_sonnet_5::ALL);
    models.extend_from_slice(claude_fable_5::ALL);
    models.extend_from_slice(claude_mythos_5::ALL);
    models.extend_from_slice(claude_opus_4_7::ALL);
    models.extend_from_slice(claude_opus_4_6::ALL);
    models.extend_from_slice(claude_opus_4_5::ALL);
    models.extend_from_slice(claude_opus_4_1::ALL);
    models.extend_from_slice(claude_sonnet_4_6::ALL);
    models.extend_from_slice(claude_sonnet_4_5::ALL);
    models.extend_from_slice(claude_opus_4::ALL);
    models.extend_from_slice(claude_sonnet_4::ALL);
    models.extend_from_slice(claude_haiku_4_5::ALL);
    models.extend_from_slice(claude_sonnet_3_7::ALL);
    models.extend_from_slice(claude_sonnet_3_5::ALL);
    models.extend_from_slice(claude_haiku_3_5::ALL);
    models.extend_from_slice(claude_haiku_3::ALL);
    models.extend_from_slice(claude_opus_3::ALL);
    models.extend_from_slice(claude_sonnet_3::ALL);
    models
}

/// Get the current recommended and limited-availability chat models.
///
/// The list is discovery metadata, never a request allowlist. Unknown IDs remain callable.
pub fn current_chat_models() -> Vec<&'static str> {
    vec![
        claude_fable_5::CLAUDE_FABLE_5,
        claude_opus_5::CLAUDE_OPUS_5,
        claude_sonnet_5::CLAUDE_SONNET_5,
        claude_haiku_4_5::CLAUDE_HAIKU_4_5,
        claude_mythos_5::CLAUDE_MYTHOS_5,
    ]
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
    model_profile(model_id)
        .is_some_and(|profile| !matches!(profile.thinking, ThinkingSupport::Unsupported))
}

/// Check if a model supports vision
pub fn supports_vision(model_id: &str) -> bool {
    model_profile(model_id).is_some_and(|profile| profile.supports_vision)
}

/// Check if a model has priority tier access
pub fn has_priority_tier(model_id: &str) -> bool {
    model_profile(model_id)
        .is_some_and(|profile| profile.supports_priority_tier == CapabilitySupport::Supported)
}

fn in_family(model_id: &str, family: &[&str]) -> bool {
    family.contains(&model_id)
}

/// Return the verified profile for a known Anthropic model ID.
///
/// Unknown IDs return `None`. Callers may still send such IDs to Anthropic; they should simply
/// avoid deriving limits or optional capabilities until the catalog is updated.
pub fn model_profile(model_id: &str) -> Option<AnthropicModelProfile> {
    if in_family(model_id, claude_opus_5::ALL) {
        return Some(current_model_profile(
            ThinkingSupport::AdaptiveOnly,
            ThinkingDisablePolicy::AllowedThroughHighEffort,
            CapabilitySupport::Unknown,
            CapabilitySupport::Unknown,
        ));
    }

    if in_family(model_id, claude_sonnet_5::ALL) {
        return Some(current_model_profile(
            ThinkingSupport::AdaptiveOnly,
            ThinkingDisablePolicy::Allowed,
            CapabilitySupport::Unsupported,
            CapabilitySupport::Unsupported,
        ));
    }

    if in_family(model_id, claude_fable_5::ALL) || in_family(model_id, claude_mythos_5::ALL) {
        return Some(current_model_profile(
            ThinkingSupport::AdaptiveOnly,
            ThinkingDisablePolicy::Forbidden,
            CapabilitySupport::Unsupported,
            CapabilitySupport::Unknown,
        ));
    }

    if in_family(model_id, claude_opus_4_7::ALL) {
        return Some(current_model_profile(
            ThinkingSupport::AdaptiveOnly,
            ThinkingDisablePolicy::Unknown,
            CapabilitySupport::Unknown,
            CapabilitySupport::Supported,
        ));
    }

    if in_family(model_id, claude_opus_4_5::ALL)
        || in_family(model_id, claude_sonnet_4_5::ALL)
        || in_family(model_id, claude_haiku_4_5::ALL)
    {
        return Some(legacy_model_profile(
            200_000,
            64_000,
            ThinkingSupport::Manual,
            true,
            true,
        ));
    }

    if in_family(model_id, claude_opus_4_6::ALL) || in_family(model_id, claude_sonnet_4_6::ALL) {
        return Some(legacy_model_profile(
            200_000,
            32_000,
            ThinkingSupport::AdaptiveAndManual,
            true,
            true,
        ));
    }

    if in_family(model_id, claude_opus_4_1::ALL)
        || in_family(model_id, claude_opus_4::ALL)
        || in_family(model_id, claude_sonnet_4::ALL)
    {
        return Some(legacy_model_profile(
            200_000,
            32_000,
            ThinkingSupport::Manual,
            true,
            true,
        ));
    }

    if in_family(model_id, claude_sonnet_3_7::ALL) {
        return Some(legacy_model_profile(
            200_000,
            64_000,
            ThinkingSupport::Manual,
            true,
            true,
        ));
    }

    if in_family(model_id, claude_sonnet_3_5::ALL) || in_family(model_id, claude_haiku_3_5::ALL) {
        return Some(legacy_model_profile(
            200_000,
            8192,
            ThinkingSupport::Unsupported,
            true,
            true,
        ));
    }

    if in_family(model_id, claude_opus_3::ALL)
        || in_family(model_id, claude_sonnet_3::ALL)
        || in_family(model_id, claude_haiku_3::ALL)
    {
        return Some(legacy_model_profile(
            200_000,
            4096,
            ThinkingSupport::Unsupported,
            true,
            false,
        ));
    }

    None
}

/// Get the context window size for a known model.
pub fn get_context_window(model_id: &str) -> Option<u32> {
    model_profile(model_id).map(|profile| profile.context_window)
}

/// Get the maximum output tokens for a known model.
pub fn get_max_output_tokens(model_id: &str) -> Option<u32> {
    try_get_max_output_tokens(model_id)
}

/// Try to get the maximum output tokens for a known model.
///
/// Vercel-aligned behavior: only known Claude models are capped. Unknown model ids
/// return `None` and should not be capped.
pub fn try_get_max_output_tokens(model_id: &str) -> Option<u32> {
    model_profile(model_id).map(|profile| profile.max_output_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn popular_recommendations_use_current_model_families() {
        assert_eq!(popular::FLAGSHIP, claude_fable_5::CLAUDE_FABLE_5);
        assert_eq!(popular::BALANCED, claude_sonnet_5::CLAUDE_SONNET_5);
        assert_eq!(popular::FAST, claude_haiku_4_5::CLAUDE_HAIKU_4_5);
        assert_eq!(popular::THINKING, claude_fable_5::CLAUDE_FABLE_5);
        assert_eq!(popular::LATEST, claude_fable_5::CLAUDE_FABLE_5);
    }

    #[test]
    fn all_chat_models_include_refreshed_and_existing_aliases() {
        let models = all_chat_models();
        assert!(models.contains(&claude_opus_5::CLAUDE_OPUS_5));
        assert!(models.contains(&claude_sonnet_5::CLAUDE_SONNET_5));
        assert!(models.contains(&claude_fable_5::CLAUDE_FABLE_5));
        assert!(models.contains(&claude_mythos_5::CLAUDE_MYTHOS_5));
        assert!(models.contains(&claude_opus_4_7::CLAUDE_OPUS_4_7));
        assert!(models.contains(&claude_sonnet_4_6::CLAUDE_SONNET_4_6));
        assert!(models.contains(&claude_opus_4_1::CLAUDE_OPUS_4_1));
        assert!(models.contains(&claude_sonnet_3_5::CLAUDE_3_5_SONNET_LATEST));
    }

    #[test]
    fn current_models_expose_distinct_verified_thinking_policy() {
        for model in [
            claude_fable_5::CLAUDE_FABLE_5,
            claude_mythos_5::CLAUDE_MYTHOS_5,
        ] {
            let profile = model_profile(model).expect("current model profile");
            assert_eq!(profile.context_window, 1_000_000);
            assert_eq!(profile.max_output_tokens, 128_000);
            assert_eq!(profile.thinking, ThinkingSupport::AdaptiveOnly);
            assert_eq!(profile.thinking_disable, ThinkingDisablePolicy::Forbidden);
            assert_eq!(profile.supports_prefill, CapabilitySupport::Unsupported);
            assert_eq!(profile.sampling, SamplingSupport::Unsupported);
        }

        let opus = model_profile(claude_opus_5::CLAUDE_OPUS_5).expect("Opus 5 profile");
        assert_eq!(
            opus.thinking_disable,
            ThinkingDisablePolicy::AllowedThroughHighEffort
        );
        assert_eq!(opus.supports_prefill, CapabilitySupport::Unknown);

        let sonnet = model_profile(claude_sonnet_5::CLAUDE_SONNET_5).expect("Sonnet 5 profile");
        assert_eq!(sonnet.thinking_disable, ThinkingDisablePolicy::Allowed);
        assert_eq!(sonnet.supports_prefill, CapabilitySupport::Unsupported);
        assert_eq!(
            sonnet.supports_priority_tier,
            CapabilitySupport::Unsupported
        );
    }

    #[test]
    fn unknown_models_are_callable_without_inferred_capabilities_or_limits() {
        let model = "claude-future-custom-model";
        assert_eq!(model_profile(model), None);
        assert_eq!(get_context_window(model), None);
        assert_eq!(get_max_output_tokens(model), None);
        assert!(!supports_thinking(model));
        assert!(!supports_vision(model));
        assert!(!has_priority_tier(model));
    }
}
