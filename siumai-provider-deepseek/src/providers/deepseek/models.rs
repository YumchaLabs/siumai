//! Curated DeepSeek language-model catalog.
//!
//! The catalog intentionally contains only model ids and capabilities verified for the native
//! DeepSeek endpoints. Unknown ids remain callable through the compatible transport, but they do
//! not receive a guessed capability profile.

use siumai_provider_openai_compatible::providers::openai_compatible::providers::models::deepseek as compat;

/// Primary official source for the curated model catalog.
pub const MODEL_CATALOG_SOURCE: &str = "https://api-docs.deepseek.com/quick_start/pricing";
/// Date on which the current-model policies were verified.
pub const MODEL_CATALOG_VERIFIED_ON: &str = "2026-08-05";

/// Native DeepSeek language-model endpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepSeekEndpoint {
    /// OpenAI-compatible chat-completions endpoint.
    ChatCompletions,
    /// Anthropic-compatible Messages endpoint.
    AnthropicMessages,
}

/// Verified capabilities for one native DeepSeek model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeepSeekModelProfile {
    /// Provider model id.
    pub id: &'static str,
    /// Native endpoints supported by this model.
    pub endpoints: &'static [DeepSeekEndpoint],
    /// Maximum accepted context length.
    pub context_window: u32,
    /// Maximum generated output length.
    pub max_output_tokens: u32,
    /// Whether provider-side thinking can be requested.
    pub supports_thinking: bool,
    /// Whether the model exposes context-cache semantics.
    pub supports_context_cache: bool,
    /// Whether native JSON output is documented.
    pub supports_json_output: bool,
    /// Whether native tool calls are documented.
    pub supports_tool_calls: bool,
    /// Whether image input is documented for this language model.
    pub supports_vision: bool,
}

impl DeepSeekModelProfile {
    /// Return whether this model is documented for the given native endpoint.
    pub fn supports_endpoint(self, endpoint: DeepSeekEndpoint) -> bool {
        self.endpoints.contains(&endpoint)
    }
}

const LANGUAGE_ENDPOINTS: &[DeepSeekEndpoint] = &[
    DeepSeekEndpoint::ChatCompletions,
    DeepSeekEndpoint::AnthropicMessages,
];

/// Current DeepSeek V4 Flash model id.
pub const DEEPSEEK_V4_FLASH: &str = "deepseek-v4-flash";
/// Current DeepSeek V4 Pro model id.
pub const DEEPSEEK_V4_PRO: &str = "deepseek-v4-pro";

/// Stable chat alias. It now points at the current fast V4 model.
pub const CHAT: &str = DEEPSEEK_V4_FLASH;
/// Stable reasoning alias. It now points at the current V4 Pro model.
pub const REASONER: &str = DEEPSEEK_V4_PRO;

/// Current native DeepSeek model profiles.
pub const PROFILES: &[DeepSeekModelProfile] = &[
    DeepSeekModelProfile {
        id: DEEPSEEK_V4_FLASH,
        endpoints: LANGUAGE_ENDPOINTS,
        context_window: 1_000_000,
        max_output_tokens: 384_000,
        supports_thinking: true,
        supports_context_cache: true,
        supports_json_output: true,
        supports_tool_calls: true,
        supports_vision: false,
    },
    DeepSeekModelProfile {
        id: DEEPSEEK_V4_PRO,
        endpoints: LANGUAGE_ENDPOINTS,
        context_window: 1_000_000,
        max_output_tokens: 384_000,
        supports_thinking: true,
        supports_context_cache: true,
        supports_json_output: true,
        supports_tool_calls: true,
        supports_vision: false,
    },
];

/// Return the verified profile for a current native model id.
pub fn profile(model: &str) -> Option<&'static DeepSeekModelProfile> {
    PROFILES.iter().find(|candidate| candidate.id == model)
}

/// Return whether a model is known to support an endpoint.
pub fn supports_endpoint(model: &str, endpoint: DeepSeekEndpoint) -> bool {
    profile(model).is_some_and(|candidate| candidate.supports_endpoint(endpoint))
}

/// Stable current DeepSeek chat-model catalog.
pub const ALL_CHAT: &[&str] = &[DEEPSEEK_V4_FLASH, DEEPSEEK_V4_PRO];
/// Stable current DeepSeek catalog.
pub const ALL: &[&str] = ALL_CHAT;

// Legacy ids remain source-compatible for callers that explicitly pin an older deployment. They
// are deliberately not advertised by `ALL` and have no inferred V4 capability profile.
#[deprecated(note = "use a current DeepSeek V4 model id instead")]
pub const DEEPSEEK_V3_0324: &str = compat::DEEPSEEK_V3_0324;
#[deprecated(note = "use a current DeepSeek V4 model id instead")]
pub const DEEPSEEK_R1_0528: &str = compat::DEEPSEEK_R1_0528;
#[deprecated(note = "use a current DeepSeek V4 model id instead")]
pub const DEEPSEEK_R1_20250120: &str = compat::DEEPSEEK_R1_20250120;
#[deprecated(note = "use a current DeepSeek V4 model id instead")]
pub const CODER: &str = compat::CODER;
#[deprecated(note = "use a current DeepSeek V4 model id instead")]
pub const DEEPSEEK_V3: &str = compat::DEEPSEEK_V3;

pub fn all_models() -> Vec<&'static str> {
    ALL.to_vec()
}

pub fn active_models() -> Vec<&'static str> {
    all_models()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_profiles_are_endpoint_specific() {
        let flash = profile(DEEPSEEK_V4_FLASH).expect("V4 Flash profile");
        assert!(flash.supports_endpoint(DeepSeekEndpoint::ChatCompletions));
        assert!(flash.supports_endpoint(DeepSeekEndpoint::AnthropicMessages));
        assert_eq!(flash.context_window, 1_000_000);
        assert_eq!(flash.max_output_tokens, 384_000);
        assert!(flash.supports_thinking);
        assert!(flash.supports_context_cache);
        assert!(flash.supports_json_output);
        assert!(flash.supports_tool_calls);
        assert!(!flash.supports_vision);

        let pro = profile(DEEPSEEK_V4_PRO).expect("V4 Pro profile");
        assert!(pro.supports_endpoint(DeepSeekEndpoint::ChatCompletions));
        assert!(pro.supports_endpoint(DeepSeekEndpoint::AnthropicMessages));
        assert!(pro.supports_thinking);
        assert!(pro.supports_context_cache);
    }

    #[test]
    fn unknown_models_do_not_receive_invented_capabilities() {
        assert!(profile("deepseek-v4-unknown").is_none());
        assert!(!supports_endpoint(
            "deepseek-v4-unknown",
            DeepSeekEndpoint::AnthropicMessages
        ));
    }

    #[test]
    fn current_catalog_excludes_legacy_ids() {
        assert_eq!(CHAT, DEEPSEEK_V4_FLASH);
        assert_eq!(REASONER, DEEPSEEK_V4_PRO);
        assert_eq!(ALL, &[DEEPSEEK_V4_FLASH, DEEPSEEK_V4_PRO]);
        assert!(!ALL.contains(&"deepseek-v3-0324"));
    }
}
