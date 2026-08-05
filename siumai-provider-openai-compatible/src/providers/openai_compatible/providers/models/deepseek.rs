//! `DeepSeek` model constants.

/// Current DeepSeek V4 Flash model.
pub const DEEPSEEK_V4_FLASH: &str = "deepseek-v4-flash";
/// Current DeepSeek V4 Pro model.
pub const DEEPSEEK_V4_PRO: &str = "deepseek-v4-pro";
/// Recommended chat model.
pub const CHAT: &str = DEEPSEEK_V4_FLASH;
/// Recommended high-capability reasoning model.
pub const REASONER: &str = DEEPSEEK_V4_PRO;

/// Retired compatibility alias for V4 Flash non-thinking mode.
pub const LEGACY_CHAT: &str = "deepseek-chat";
/// Retired compatibility alias for V4 Flash thinking mode.
pub const LEGACY_REASONER: &str = "deepseek-reasoner";

// Specific model versions
/// `DeepSeek` V3 (2024-03-24)
pub const DEEPSEEK_V3_0324: &str = "deepseek-v3-0324";
/// `DeepSeek` R1 (2025-05-28)
pub const DEEPSEEK_R1_0528: &str = "deepseek-r1-0528";
/// `DeepSeek` R1 (2025-01-20)
pub const DEEPSEEK_R1_20250120: &str = "deepseek-r1-20250120";

// Legacy models (deprecated)
/// `DeepSeek` Coder model (legacy)
pub const CODER: &str = "deepseek-coder";
/// `DeepSeek` V3 model (legacy alias)
pub const DEEPSEEK_V3: &str = "deepseek-v3";

/// All `DeepSeek` models
pub const ALL: &[&str] = &[CHAT, REASONER];

/// Get all `DeepSeek` models
pub fn all_models() -> Vec<String> {
    ALL.iter().map(|&s| s.to_string()).collect()
}

/// Get current active models (non-legacy)
pub fn active_models() -> Vec<String> {
    all_models()
}
