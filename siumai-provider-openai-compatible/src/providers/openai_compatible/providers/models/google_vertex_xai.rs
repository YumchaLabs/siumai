//! Google Vertex xAI model constants.

/// Google Vertex xAI chat/language-model constants.
pub mod chat {
    pub const GROK_4_20_REASONING: &str = "xai/grok-4.20-reasoning";
    pub const GROK_4_20_NON_REASONING: &str = "xai/grok-4.20-non-reasoning";
    pub const GROK_4_1_FAST_REASONING: &str = "xai/grok-4.1-fast-reasoning";
    pub const GROK_4_1_FAST_NON_REASONING: &str = "xai/grok-4.1-fast-non-reasoning";
}

pub const CHAT: &str = chat::GROK_4_1_FAST_REASONING;

pub const ALL_CHAT: &[&str] = &[
    chat::GROK_4_20_REASONING,
    chat::GROK_4_20_NON_REASONING,
    chat::GROK_4_1_FAST_REASONING,
    chat::GROK_4_1_FAST_NON_REASONING,
];

pub fn all_models() -> Vec<String> {
    ALL_CHAT.iter().map(|model| (*model).to_string()).collect()
}
