//! Curated Anthropic-on-Vertex model constants aligned with the audited AI SDK public subset.

/// AI SDK-style model-id carrier for Google Vertex Anthropic messages models.
pub type GoogleVertexAnthropicMessagesModelId = String;

/// Anthropic-on-Vertex chat/language-model constants.
pub mod chat {
    pub const CLAUDE_OPUS_4_7: &str = "claude-opus-4-7";
    pub const CLAUDE_OPUS_4_6: &str = "claude-opus-4-6";
    pub const CLAUDE_SONNET_4_6: &str = "claude-sonnet-4-6";
    pub const CLAUDE_OPUS_4_5_AT_20251101: &str = "claude-opus-4-5@20251101";
    pub const CLAUDE_SONNET_4_5_AT_20250929: &str = "claude-sonnet-4-5@20250929";
    pub const CLAUDE_OPUS_4_1_AT_20250805: &str = "claude-opus-4-1@20250805";
    pub const CLAUDE_OPUS_4_AT_20250514: &str = "claude-opus-4@20250514";
    pub const CLAUDE_SONNET_4_AT_20250514: &str = "claude-sonnet-4@20250514";
    pub const CLAUDE_3_7_SONNET_AT_20250219: &str = "claude-3-7-sonnet@20250219";
    pub const CLAUDE_3_5_SONNET_V2_AT_20241022: &str = "claude-3-5-sonnet-v2@20241022";
    pub const CLAUDE_3_5_HAIKU_AT_20241022: &str = "claude-3-5-haiku@20241022";
    pub const CLAUDE_3_5_SONNET_AT_20240620: &str = "claude-3-5-sonnet@20240620";
    pub const CLAUDE_3_HAIKU_AT_20240307: &str = "claude-3-haiku@20240307";
    pub const CLAUDE_3_SONNET_AT_20240229: &str = "claude-3-sonnet@20240229";
    pub const CLAUDE_3_OPUS_AT_20240229: &str = "claude-3-opus@20240229";
}

/// Curated default chat model for the public Rust convenience surface.
///
/// Upstream `createVertexAnthropic()` does not choose a model automatically; Rust still keeps one
/// representative constant for grouped model modules and docs.
pub const CHAT: &str = chat::CLAUDE_SONNET_4_6;

pub const ALL_CHAT: &[&str] = &[
    chat::CLAUDE_OPUS_4_7,
    chat::CLAUDE_OPUS_4_6,
    chat::CLAUDE_SONNET_4_6,
    chat::CLAUDE_OPUS_4_5_AT_20251101,
    chat::CLAUDE_SONNET_4_5_AT_20250929,
    chat::CLAUDE_OPUS_4_1_AT_20250805,
    chat::CLAUDE_OPUS_4_AT_20250514,
    chat::CLAUDE_SONNET_4_AT_20250514,
    chat::CLAUDE_3_7_SONNET_AT_20250219,
    chat::CLAUDE_3_5_SONNET_V2_AT_20241022,
    chat::CLAUDE_3_5_HAIKU_AT_20241022,
    chat::CLAUDE_3_5_SONNET_AT_20240620,
    chat::CLAUDE_3_HAIKU_AT_20240307,
    chat::CLAUDE_3_SONNET_AT_20240229,
    chat::CLAUDE_3_OPUS_AT_20240229,
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn curated_default_matches_current_public_representative() {
        assert_eq!(CHAT, chat::CLAUDE_SONNET_4_6);
    }

    #[test]
    fn curated_list_includes_primary_default() {
        assert!(ALL_CHAT.contains(&CHAT));
    }

    #[test]
    fn curated_subset_matches_audited_upstream_model_id_union() {
        let expected = BTreeSet::from([
            "claude-opus-4-7",
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-opus-4-5@20251101",
            "claude-sonnet-4-5@20250929",
            "claude-opus-4-1@20250805",
            "claude-opus-4@20250514",
            "claude-sonnet-4@20250514",
            "claude-3-7-sonnet@20250219",
            "claude-3-5-sonnet-v2@20241022",
            "claude-3-5-haiku@20241022",
            "claude-3-5-sonnet@20240620",
            "claude-3-haiku@20240307",
            "claude-3-sonnet@20240229",
            "claude-3-opus@20240229",
        ]);
        let actual = ALL_CHAT.iter().copied().collect::<BTreeSet<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn model_id_type_alias_accepts_string_values() {
        let _: GoogleVertexAnthropicMessagesModelId = chat::CLAUDE_SONNET_4_6.to_string();
    }
}
