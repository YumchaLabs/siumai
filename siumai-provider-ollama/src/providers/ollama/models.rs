//! Curated Ollama model constants for the public provider surface.
//!
//! These constants intentionally model the current curated/default subset that Siumai advertises
//! through the public facade and provider catalog. They are narrower than the legacy
//! `model_constants` module, which still keeps broader aliases and historical ids.

/// Ollama chat/language-model constants.
pub mod chat {
    pub const LLAMA_3_2_LATEST: &str = "llama3.2:latest";
    pub const LLAMA_3_2_3B: &str = "llama3.2:3b";
    pub const LLAMA_3_2_1B: &str = "llama3.2:1b";
    pub const LLAMA_3_1_LATEST: &str = "llama3.1:latest";
    pub const LLAMA_3_1_8B: &str = "llama3.1:8b";
    pub const LLAMA_3_1_70B: &str = "llama3.1:70b";
    pub const MISTRAL_LATEST: &str = "mistral:latest";
    pub const MISTRAL_7B: &str = "mistral:7b";
    pub const CODE_LLAMA_LATEST: &str = "codellama:latest";
    pub const CODE_LLAMA_7B: &str = "codellama:7b";
    pub const CODE_LLAMA_13B: &str = "codellama:13b";
    pub const CODE_LLAMA_34B: &str = "codellama:34b";
    pub const PHI_3_LATEST: &str = "phi3:latest";
    pub const PHI_3_MINI: &str = "phi3:mini";
    pub const PHI_3_MEDIUM: &str = "phi3:medium";
    pub const GEMMA_LATEST: &str = "gemma:latest";
    pub const GEMMA_2B: &str = "gemma:2b";
    pub const GEMMA_7B: &str = "gemma:7b";
    pub const QWEN2_LATEST: &str = "qwen2:latest";
    pub const QWEN2_0_5B: &str = "qwen2:0.5b";
    pub const QWEN2_1_5B: &str = "qwen2:1.5b";
    pub const QWEN2_7B: &str = "qwen2:7b";
    pub const QWEN2_72B: &str = "qwen2:72b";
    pub const DEEPSEEK_CODER_LATEST: &str = "deepseek-coder:latest";
    pub const DEEPSEEK_CODER_6_7B: &str = "deepseek-coder:6.7b";
    pub const DEEPSEEK_CODER_33B: &str = "deepseek-coder:33b";
}

/// Ollama embedding-model constants.
pub mod embedding {
    pub const NOMIC_EMBED_TEXT: &str = "nomic-embed-text:latest";
    pub const ALL_MINILM: &str = "all-minilm:latest";
}

pub const CHAT: &str = chat::LLAMA_3_2_LATEST;
pub const EMBEDDING: &str = embedding::NOMIC_EMBED_TEXT;

pub const ALL_CHAT: &[&str] = &[
    chat::LLAMA_3_2_LATEST,
    chat::LLAMA_3_2_3B,
    chat::LLAMA_3_2_1B,
    chat::LLAMA_3_1_LATEST,
    chat::LLAMA_3_1_8B,
    chat::LLAMA_3_1_70B,
    chat::MISTRAL_LATEST,
    chat::MISTRAL_7B,
    chat::CODE_LLAMA_LATEST,
    chat::CODE_LLAMA_7B,
    chat::CODE_LLAMA_13B,
    chat::CODE_LLAMA_34B,
    chat::PHI_3_LATEST,
    chat::PHI_3_MINI,
    chat::PHI_3_MEDIUM,
    chat::GEMMA_LATEST,
    chat::GEMMA_2B,
    chat::GEMMA_7B,
    chat::QWEN2_LATEST,
    chat::QWEN2_0_5B,
    chat::QWEN2_1_5B,
    chat::QWEN2_7B,
    chat::QWEN2_72B,
    chat::DEEPSEEK_CODER_LATEST,
    chat::DEEPSEEK_CODER_6_7B,
    chat::DEEPSEEK_CODER_33B,
];

pub const ALL_EMBEDDING: &[&str] = &[embedding::NOMIC_EMBED_TEXT, embedding::ALL_MINILM];

pub fn all_models() -> Vec<&'static str> {
    let mut models = Vec::with_capacity(ALL_CHAT.len() + ALL_EMBEDDING.len());
    models.extend_from_slice(ALL_CHAT);
    models.extend_from_slice(ALL_EMBEDDING);
    models
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curated_defaults_match_current_runtime_defaults() {
        assert_eq!(CHAT, chat::LLAMA_3_2_LATEST);
        assert_eq!(EMBEDDING, embedding::NOMIC_EMBED_TEXT);
    }

    #[test]
    fn curated_lists_cover_primary_defaults() {
        assert!(ALL_CHAT.contains(&CHAT));
        assert!(ALL_EMBEDDING.contains(&EMBEDDING));
        assert!(all_models().contains(&CHAT));
        assert!(all_models().contains(&EMBEDDING));
    }
}
