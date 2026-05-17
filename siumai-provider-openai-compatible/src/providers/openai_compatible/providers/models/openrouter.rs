//! `OpenRouter` model constants
/// `OpenAI` models via `OpenRouter`
pub mod openai {
    pub const GPT_4: &str = "openai/gpt-4";
    pub const GPT_4_TURBO: &str = "openai/gpt-4-turbo";
    pub const GPT_4O: &str = "openai/gpt-4o";
    pub const GPT_4O_MINI: &str = "openai/gpt-4o-mini";
    pub const GPT_4_1: &str = "openai/gpt-4.1";
    pub const GPT_4_1_MINI: &str = "openai/gpt-4.1-mini";
    pub const O1: &str = "openai/o1";
    pub const O1_MINI: &str = "openai/o1-mini";
    pub const O3_MINI: &str = "openai/o3-mini";
}

/// Anthropic models via `OpenRouter`
pub mod anthropic {
    pub const CLAUDE_3_5_SONNET: &str = "anthropic/claude-3.5-sonnet";
    pub const CLAUDE_3_5_HAIKU: &str = "anthropic/claude-3.5-haiku";
    pub const CLAUDE_SONNET_4: &str = "anthropic/claude-sonnet-4";
    pub const CLAUDE_OPUS_4: &str = "anthropic/claude-opus-4";
    pub const CLAUDE_OPUS_4_1: &str = "anthropic/claude-opus-4.1";
}

/// Google models via `OpenRouter`
pub mod google {
    pub const GEMINI_PRO: &str = "google/gemini-pro";
    pub const GEMINI_1_5_PRO: &str = "google/gemini-1.5-pro";
    pub const GEMINI_2_0_FLASH: &str = "google/gemini-2.0-flash";
    pub const GEMINI_2_5_FLASH: &str = "google/gemini-2.5-flash";
    pub const GEMINI_2_5_PRO: &str = "google/gemini-2.5-pro";
}

/// DeepSeek models via `OpenRouter`
pub mod deepseek {
    pub const DEEPSEEK_CHAT: &str = "deepseek/deepseek-chat";
    pub const DEEPSEEK_REASONER: &str = "deepseek/deepseek-reasoner";
    pub const DEEPSEEK_V3: &str = "deepseek/deepseek-v3";
    pub const DEEPSEEK_R1: &str = "deepseek/deepseek-r1";
}

/// Meta models via `OpenRouter`
pub mod meta {
    pub const LLAMA_3_1_8B: &str = "meta-llama/llama-3.1-8b-instruct";
    pub const LLAMA_3_1_70B: &str = "meta-llama/llama-3.1-70b-instruct";
    pub const LLAMA_3_1_405B: &str = "meta-llama/llama-3.1-405b-instruct";
    pub const LLAMA_3_2_1B: &str = "meta-llama/llama-3.2-1b-instruct";
    pub const LLAMA_3_2_3B: &str = "meta-llama/llama-3.2-3b-instruct";
}

/// Mistral models via `OpenRouter`
pub mod mistral {
    pub const MISTRAL_7B: &str = "mistralai/mistral-7b-instruct";
    pub const MIXTRAL_8X7B: &str = "mistralai/mixtral-8x7b-instruct";
    pub const MIXTRAL_8X22B: &str = "mistralai/mixtral-8x22b-instruct";
    pub const MISTRAL_LARGE: &str = "mistralai/mistral-large";
}

/// Popular models collection
pub mod popular {
    use super::*;

    pub const GPT_4O: &str = openai::GPT_4O;
    pub const GPT_4_1: &str = openai::GPT_4_1;
    pub const CLAUDE_OPUS_4_1: &str = anthropic::CLAUDE_OPUS_4_1;
    pub const CLAUDE_SONNET_4: &str = anthropic::CLAUDE_SONNET_4;
    pub const GEMINI_2_5_PRO: &str = google::GEMINI_2_5_PRO;
    pub const DEEPSEEK_REASONER: &str = deepseek::DEEPSEEK_REASONER;
    pub const LLAMA_3_1_405B: &str = meta::LLAMA_3_1_405B;
}

/// Get all `OpenRouter` models
pub fn all_models() -> Vec<String> {
    let mut models = Vec::new();

    // OpenAI models
    models.extend_from_slice(&[
        openai::GPT_4.to_string(),
        openai::GPT_4_TURBO.to_string(),
        openai::GPT_4O.to_string(),
        openai::GPT_4O_MINI.to_string(),
        openai::GPT_4_1.to_string(),
        openai::GPT_4_1_MINI.to_string(),
        openai::O1.to_string(),
        openai::O1_MINI.to_string(),
        openai::O3_MINI.to_string(),
    ]);

    // Anthropic models
    models.extend_from_slice(&[
        anthropic::CLAUDE_3_5_SONNET.to_string(),
        anthropic::CLAUDE_3_5_HAIKU.to_string(),
        anthropic::CLAUDE_SONNET_4.to_string(),
        anthropic::CLAUDE_OPUS_4.to_string(),
        anthropic::CLAUDE_OPUS_4_1.to_string(),
    ]);

    // Google models
    models.extend_from_slice(&[
        google::GEMINI_PRO.to_string(),
        google::GEMINI_1_5_PRO.to_string(),
        google::GEMINI_2_0_FLASH.to_string(),
        google::GEMINI_2_5_FLASH.to_string(),
        google::GEMINI_2_5_PRO.to_string(),
    ]);

    // DeepSeek models
    models.extend_from_slice(&[
        deepseek::DEEPSEEK_CHAT.to_string(),
        deepseek::DEEPSEEK_REASONER.to_string(),
        deepseek::DEEPSEEK_V3.to_string(),
        deepseek::DEEPSEEK_R1.to_string(),
    ]);

    // Meta models
    models.extend_from_slice(&[
        meta::LLAMA_3_1_8B.to_string(),
        meta::LLAMA_3_1_70B.to_string(),
        meta::LLAMA_3_1_405B.to_string(),
        meta::LLAMA_3_2_1B.to_string(),
        meta::LLAMA_3_2_3B.to_string(),
    ]);

    // Mistral models
    models.extend_from_slice(&[
        mistral::MISTRAL_7B.to_string(),
        mistral::MIXTRAL_8X7B.to_string(),
        mistral::MIXTRAL_8X22B.to_string(),
        mistral::MISTRAL_LARGE.to_string(),
    ]);

    models
}
