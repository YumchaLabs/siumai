//! Dated DeepSeek model advisories.
//!
//! Model identifiers remain open input. These constants improve discovery and provide exact-match
//! policy evidence; they are not an allowlist and do not select a host default.

/// Official source used for the current DeepSeek model advisory catalog.
pub const MODEL_CATALOG_SOURCE: &str = "https://api-docs.deepseek.com/quick_start/pricing";
/// Date of the latest model-ID advisory verification, independent of protocol evidence.
pub const MODEL_CATALOG_VERIFIED_ON: &str = "2026-10-25";
/// Official source used for the current DeepSeek Responses model advisory.
pub const RESPONSES_CATALOG_SOURCE: &str = "https://api-docs.deepseek.com/guides/responses_api/";
/// Date on which the Responses model advisory was verified.
pub const RESPONSES_CATALOG_VERIFIED_ON: &str = "2026-10-25";

/// Current DeepSeek Flash model identifier.
pub const DEEPSEEK_FLASH: &str = "deepseek-flash";
/// Legacy V4 Flash identifier retained for provider-side compatibility routing.
pub const DEEPSEEK_V4_FLASH: &str = "deepseek-v4-flash";
/// Current DeepSeek V4 Pro model identifier.
pub const DEEPSEEK_V4_PRO: &str = "deepseek-v4-pro";

/// Models verified for Chat Completions on the public DeepSeek API.
pub const ALL_CHAT: &[&str] = &[DEEPSEEK_FLASH, DEEPSEEK_V4_FLASH, DEEPSEEK_V4_PRO];
/// Models verified for the public DeepSeek Responses API.
pub const ALL_RESPONSES: &[&str] = &[DEEPSEEK_FLASH, DEEPSEEK_V4_FLASH, DEEPSEEK_V4_PRO];
/// All model-ID hints, including retained compatibility identifiers.
pub const ALL: &[&str] = ALL_CHAT;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_preserves_current_and_legacy_ids() {
        assert_eq!(DEEPSEEK_FLASH, "deepseek-flash");
        assert_eq!(DEEPSEEK_V4_FLASH, "deepseek-v4-flash");
        assert_eq!(
            ALL_CHAT,
            &[DEEPSEEK_FLASH, DEEPSEEK_V4_FLASH, DEEPSEEK_V4_PRO]
        );
        assert_eq!(ALL_RESPONSES, ALL_CHAT);
        assert!(!ALL.contains(&"deepseek-v3-0324"));
    }
}
