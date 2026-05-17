//! `DeepSeek` model constants
/// `DeepSeek` Chat model (points to DeepSeek-V3-0324)
pub const CHAT: &str = "deepseek-chat";
/// `DeepSeek` Reasoner model (points to DeepSeek-R1-0528)
pub const REASONER: &str = "deepseek-reasoner";

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
pub const ALL: &[&str] = &[
    CHAT,
    REASONER,
    DEEPSEEK_V3_0324,
    DEEPSEEK_R1_0528,
    DEEPSEEK_R1_20250120,
    CODER,
    DEEPSEEK_V3,
];

/// Get all `DeepSeek` models
pub fn all_models() -> Vec<String> {
    ALL.iter().map(|&s| s.to_string()).collect()
}

/// Get current active models (non-legacy)
pub fn active_models() -> Vec<String> {
    vec![
        CHAT.to_string(),
        REASONER.to_string(),
        DEEPSEEK_V3_0324.to_string(),
        DEEPSEEK_R1_0528.to_string(),
        DEEPSEEK_R1_20250120.to_string(),
    ]
}
