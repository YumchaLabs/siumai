//! Perplexity model constants aligned with the audited AI SDK package subset.
/// Perplexity chat/language-model constants.
pub mod chat {
    pub const SONAR_DEEP_RESEARCH: &str = "sonar-deep-research";
    pub const SONAR_REASONING_PRO: &str = "sonar-reasoning-pro";
    pub const SONAR_REASONING: &str = "sonar-reasoning";
    pub const SONAR_PRO: &str = "sonar-pro";
    pub const SONAR: &str = "sonar";
}

pub const CHAT: &str = chat::SONAR;

pub const ALL_CHAT: &[&str] = &[
    chat::SONAR_DEEP_RESEARCH,
    chat::SONAR_REASONING_PRO,
    chat::SONAR_REASONING,
    chat::SONAR_PRO,
    chat::SONAR,
];
