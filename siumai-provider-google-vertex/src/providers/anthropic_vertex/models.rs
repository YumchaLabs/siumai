//! Curated Anthropic-on-Vertex model constants aligned with the audited public subset.

/// Anthropic-on-Vertex chat/language-model constants.
pub mod chat {
    pub const CLAUDE_SONNET_4_5_LATEST: &str = "claude-sonnet-4-5-latest";
    pub const CLAUDE_3_7_SONNET_20250219: &str = "claude-3-7-sonnet-20250219";
    pub const CLAUDE_3_5_SONNET_20241022: &str = "claude-3-5-sonnet-20241022";
}

pub const CHAT: &str = chat::CLAUDE_SONNET_4_5_LATEST;

pub const ALL_CHAT: &[&str] = &[
    chat::CLAUDE_SONNET_4_5_LATEST,
    chat::CLAUDE_3_7_SONNET_20250219,
    chat::CLAUDE_3_5_SONNET_20241022,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curated_default_matches_current_runtime_default() {
        assert_eq!(CHAT, chat::CLAUDE_SONNET_4_5_LATEST);
    }

    #[test]
    fn curated_list_includes_primary_default() {
        assert!(ALL_CHAT.contains(&CHAT));
    }
}
