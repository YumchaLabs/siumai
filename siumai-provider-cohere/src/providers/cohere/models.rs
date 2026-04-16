//! Curated Cohere model constants aligned with the audited AI SDK package subset.

/// Cohere chat/language-model constants.
pub mod chat {
    pub const COMMAND_A_03_2025: &str = "command-a-03-2025";
    pub const COMMAND_A_REASONING_08_2025: &str = "command-a-reasoning-08-2025";
    pub const COMMAND_R7B_12_2024: &str = "command-r7b-12-2024";
    pub const COMMAND_R_PLUS_04_2024: &str = "command-r-plus-04-2024";
    pub const COMMAND_R_PLUS: &str = "command-r-plus";
    pub const COMMAND_R_08_2024: &str = "command-r-08-2024";
    pub const COMMAND_R_03_2024: &str = "command-r-03-2024";
    pub const COMMAND_R: &str = "command-r";
    pub const COMMAND: &str = "command";
    pub const COMMAND_NIGHTLY: &str = "command-nightly";
    pub const COMMAND_LIGHT: &str = "command-light";
    pub const COMMAND_LIGHT_NIGHTLY: &str = "command-light-nightly";
}

/// Cohere embedding-model constants.
pub mod embedding {
    pub const EMBED_ENGLISH_V3: &str = "embed-english-v3.0";
    pub const EMBED_MULTILINGUAL_V3: &str = "embed-multilingual-v3.0";
    pub const EMBED_ENGLISH_LIGHT_V3: &str = "embed-english-light-v3.0";
    pub const EMBED_MULTILINGUAL_LIGHT_V3: &str = "embed-multilingual-light-v3.0";
    pub const EMBED_V4: &str = "embed-v4.0";
}

/// Cohere rerank-model constants.
pub mod rerank {
    pub const RERANK_V3_5: &str = "rerank-v3.5";
    pub const RERANK_ENGLISH_V3: &str = "rerank-english-v3.0";
    pub const RERANK_MULTILINGUAL_V3: &str = "rerank-multilingual-v3.0";
}

pub const CHAT: &str = chat::COMMAND_A_03_2025;
pub const EMBEDDING: &str = embedding::EMBED_V4;
pub const RERANK: &str = rerank::RERANK_V3_5;

pub const ALL_CHAT: &[&str] = &[
    chat::COMMAND_A_03_2025,
    chat::COMMAND_A_REASONING_08_2025,
    chat::COMMAND_R7B_12_2024,
    chat::COMMAND_R_PLUS_04_2024,
    chat::COMMAND_R_PLUS,
    chat::COMMAND_R_08_2024,
    chat::COMMAND_R_03_2024,
    chat::COMMAND_R,
    chat::COMMAND,
    chat::COMMAND_NIGHTLY,
    chat::COMMAND_LIGHT,
    chat::COMMAND_LIGHT_NIGHTLY,
];

pub const ALL_EMBEDDING: &[&str] = &[
    embedding::EMBED_ENGLISH_V3,
    embedding::EMBED_MULTILINGUAL_V3,
    embedding::EMBED_ENGLISH_LIGHT_V3,
    embedding::EMBED_MULTILINGUAL_LIGHT_V3,
    embedding::EMBED_V4,
];

pub const ALL_RERANK: &[&str] = &[
    rerank::RERANK_V3_5,
    rerank::RERANK_ENGLISH_V3,
    rerank::RERANK_MULTILINGUAL_V3,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curated_defaults_match_current_runtime_defaults() {
        assert_eq!(CHAT, chat::COMMAND_A_03_2025);
        assert_eq!(EMBEDDING, embedding::EMBED_V4);
        assert_eq!(RERANK, rerank::RERANK_V3_5);
    }

    #[test]
    fn curated_lists_include_primary_defaults() {
        assert!(ALL_CHAT.contains(&CHAT));
        assert!(ALL_EMBEDDING.contains(&EMBEDDING));
        assert!(ALL_RERANK.contains(&RERANK));
    }
}
