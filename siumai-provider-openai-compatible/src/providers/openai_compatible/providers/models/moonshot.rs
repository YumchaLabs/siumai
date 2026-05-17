//! MoonshotAI model constants aligned with the audited AI SDK package subset.
/// Kimi K2.
pub const KIMI_K2: &str = "kimi-k2";

/// Kimi K2 0905.
pub const KIMI_K2_0905: &str = "kimi-k2-0905";

/// Kimi K2 Thinking.
pub const KIMI_K2_THINKING: &str = "kimi-k2-thinking";

/// Kimi K2 Thinking Turbo.
pub const KIMI_K2_THINKING_TURBO: &str = "kimi-k2-thinking-turbo";

/// Kimi K2 Turbo.
pub const KIMI_K2_TURBO: &str = "kimi-k2-turbo";

/// Kimi K2.5.
pub const KIMI_K2P5: &str = "kimi-k2.5";

/// Kimi Latest - Auto-updated to the latest Kimi model
pub const KIMI_LATEST: &str = "kimi-latest";

/// Moonshot V1 Auto - Auto-updated to the latest V1 model
pub const MOONSHOT_V1_AUTO: &str = "moonshot-v1-auto";

/// Moonshot V1 8K - Standard context window (8,192 tokens)
/// Optimized for short text generation tasks
pub const MOONSHOT_V1_8K: &str = "moonshot-v1-8k";

/// Moonshot V1 32K - Medium context window (32,768 tokens)
/// Suitable for long documents and complex conversations
pub const MOONSHOT_V1_32K: &str = "moonshot-v1-32k";

/// Moonshot V1 128K - Large context window (128,000 tokens)
/// Ideal for research, academic work, and large document generation
pub const MOONSHOT_V1_128K: &str = "moonshot-v1-128k";

/// Moonshot V1 8K Vision Preview - Vision model with 8K context
pub const MOONSHOT_V1_8K_VISION_PREVIEW: &str = "moonshot-v1-8k-vision-preview";

/// Moonshot V1 32K Vision Preview - Vision model with 32K context
pub const MOONSHOT_V1_32K_VISION_PREVIEW: &str = "moonshot-v1-32k-vision-preview";

/// Moonshot V1 128K Vision Preview - Vision model with 128K context
pub const MOONSHOT_V1_128K_VISION_PREVIEW: &str = "moonshot-v1-128k-vision-preview";

/// All Moonshot chat models
pub const ALL_CHAT: &[&str] = &[
    KIMI_K2,
    KIMI_K2_0905,
    KIMI_K2_THINKING,
    KIMI_K2_THINKING_TURBO,
    KIMI_K2_TURBO,
    KIMI_K2P5,
    KIMI_LATEST,
    MOONSHOT_V1_AUTO,
    MOONSHOT_V1_8K,
    MOONSHOT_V1_32K,
    MOONSHOT_V1_128K,
];

/// All Moonshot vision models
pub const ALL_VISION: &[&str] = &[
    MOONSHOT_V1_8K_VISION_PREVIEW,
    MOONSHOT_V1_32K_VISION_PREVIEW,
    MOONSHOT_V1_128K_VISION_PREVIEW,
];

/// Get all Moonshot models
pub fn all_models() -> Vec<String> {
    let mut models = Vec::new();
    models.extend(ALL_CHAT.iter().map(|&s| s.to_string()));
    models.extend(ALL_VISION.iter().map(|&s| s.to_string()));
    models
}

/// Get recommended model for different use cases
pub mod recommended {
    use super::*;

    /// Best for general chat with latest features
    pub const CHAT: &str = KIMI_K2_0905;

    /// Best for long-context processing
    pub const LONG_CONTEXT: &str = MOONSHOT_V1_128K;

    /// Best for cost-effective short conversations
    pub const COST_EFFECTIVE: &str = MOONSHOT_V1_8K;

    /// Best for vision tasks
    pub const VISION: &str = MOONSHOT_V1_128K_VISION_PREVIEW;
}
