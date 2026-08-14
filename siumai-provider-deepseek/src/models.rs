//! Dated DeepSeek model advisories.
//!
//! Model identifiers remain open input. These constants improve discovery and provide exact-match
//! policy evidence; they are not an allowlist and do not select a host default.

/// Official source used for the current DeepSeek model advisory catalog.
pub const MODEL_CATALOG_SOURCE: &str = "https://api-docs.deepseek.com/quick_start/pricing";
/// Date on which the advisory catalog was verified.
pub const MODEL_CATALOG_VERIFIED_ON: &str = "2026-08-05";
/// Official source used for the current DeepSeek Responses model advisory.
pub const RESPONSES_CATALOG_SOURCE: &str = "https://api-docs.deepseek.com/guides/responses_api/";
/// Date on which the Responses model advisory was verified.
pub const RESPONSES_CATALOG_VERIFIED_ON: &str = "2026-08-14";

/// Current DeepSeek V4 Flash model identifier.
pub const DEEPSEEK_V4_FLASH: &str = "deepseek-v4-flash";
/// Current DeepSeek V4 Pro model identifier.
pub const DEEPSEEK_V4_PRO: &str = "deepseek-v4-pro";

/// Models verified for Chat Completions on the public DeepSeek API.
pub const ALL_CHAT: &[&str] = &[DEEPSEEK_V4_FLASH, DEEPSEEK_V4_PRO];
/// Models verified for the public DeepSeek Responses API.
pub const ALL_RESPONSES: &[&str] = &[DEEPSEEK_V4_FLASH, DEEPSEEK_V4_PRO];
/// All current exact model identifiers represented by this advisory catalog.
pub const ALL: &[&str] = ALL_CHAT;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_contains_only_exact_current_ids() {
        assert_eq!(ALL_CHAT, &[DEEPSEEK_V4_FLASH, DEEPSEEK_V4_PRO]);
        assert_eq!(ALL_RESPONSES, &[DEEPSEEK_V4_FLASH, DEEPSEEK_V4_PRO]);
        assert!(!ALL.contains(&"deepseek-v3-0324"));
    }
}
