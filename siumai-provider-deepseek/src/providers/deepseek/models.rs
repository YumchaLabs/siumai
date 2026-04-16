//! Curated DeepSeek model constants aligned with the audited AI SDK package subset.
//!
//! AI SDK's native DeepSeek package exposes a chat-only provider surface with stable model ids
//! `deepseek-chat` and `deepseek-reasoner`. Siumai keeps those stable ids as the canonical public
//! family surface while still preserving access to a few provider-known versioned aliases.

use siumai_provider_openai_compatible::providers::openai_compatible::providers::models::deepseek as compat;

/// Stable DeepSeek chat/language-model constants.
pub mod chat {
    pub const DEEPSEEK_CHAT: &str =
        siumai_provider_openai_compatible::providers::openai_compatible::providers::models::deepseek::CHAT;
    pub const DEEPSEEK_REASONER: &str =
        siumai_provider_openai_compatible::providers::openai_compatible::providers::models::deepseek::REASONER;
}

pub const CHAT: &str = chat::DEEPSEEK_CHAT;
pub const REASONER: &str = chat::DEEPSEEK_REASONER;

pub const DEEPSEEK_V3_0324: &str = compat::DEEPSEEK_V3_0324;
pub const DEEPSEEK_R1_0528: &str = compat::DEEPSEEK_R1_0528;
pub const DEEPSEEK_R1_20250120: &str = compat::DEEPSEEK_R1_20250120;
pub const CODER: &str = compat::CODER;
pub const DEEPSEEK_V3: &str = compat::DEEPSEEK_V3;

pub const ALL_CHAT: &[&str] = &[CHAT, REASONER];
pub const ALL: &[&str] = &[
    CHAT,
    REASONER,
    DEEPSEEK_V3_0324,
    DEEPSEEK_R1_0528,
    DEEPSEEK_R1_20250120,
    CODER,
    DEEPSEEK_V3,
];

pub fn all_models() -> Vec<&'static str> {
    ALL.to_vec()
}

pub fn active_models() -> Vec<&'static str> {
    vec![
        CHAT,
        REASONER,
        DEEPSEEK_V3_0324,
        DEEPSEEK_R1_0528,
        DEEPSEEK_R1_20250120,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curated_chat_surface_matches_ai_sdk_subset() {
        assert_eq!(CHAT, compat::CHAT);
        assert_eq!(REASONER, compat::REASONER);
        assert_eq!(ALL_CHAT, &[compat::CHAT, compat::REASONER]);
    }
}
