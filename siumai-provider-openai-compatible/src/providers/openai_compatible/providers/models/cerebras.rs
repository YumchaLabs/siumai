//! Cerebras model constants aligned with the audited AI SDK package subset.

/// Cerebras chat/language-model constants.
pub mod chat {
    /// Production models.
    pub const LLAMA3_1_8B: &str = "llama3.1-8b";
    pub const GPT_OSS_120B: &str = "gpt-oss-120b";

    /// Preview models.
    pub const QWEN_3_235B_A22B_INSTRUCT_2507: &str = "qwen-3-235b-a22b-instruct-2507";
    pub const QWEN_3_235B_A22B_THINKING_2507: &str = "qwen-3-235b-a22b-thinking-2507";
    pub const ZAI_GLM_4_6: &str = "zai-glm-4.6";
    pub const ZAI_GLM_4_7: &str = "zai-glm-4.7";
}

pub const CHAT: &str = chat::LLAMA3_1_8B;

pub const ALL_CHAT: &[&str] = &[
    chat::LLAMA3_1_8B,
    chat::GPT_OSS_120B,
    chat::QWEN_3_235B_A22B_INSTRUCT_2507,
    chat::QWEN_3_235B_A22B_THINKING_2507,
    chat::ZAI_GLM_4_6,
    chat::ZAI_GLM_4_7,
];

/// Get all curated Cerebras chat models from the audited AI SDK subset.
pub fn all_models() -> Vec<String> {
    ALL_CHAT.iter().map(|&model| model.to_string()).collect()
}
