//! Mistral model constants aligned with the audited AI SDK package subset.
/// Mistral chat/language-model constants.
pub mod chat {
    pub const MINISTRAL_3B_LATEST: &str = "ministral-3b-latest";
    pub const MINISTRAL_8B_LATEST: &str = "ministral-8b-latest";
    pub const MINISTRAL_14B_LATEST: &str = "ministral-14b-latest";
    pub const MISTRAL_LARGE_LATEST: &str = "mistral-large-latest";
    pub const MISTRAL_MEDIUM_LATEST: &str = "mistral-medium-latest";
    pub const MISTRAL_LARGE_2512: &str = "mistral-large-2512";
    pub const MISTRAL_MEDIUM_2508: &str = "mistral-medium-2508";
    pub const MISTRAL_MEDIUM_2505: &str = "mistral-medium-2505";
    pub const MISTRAL_SMALL_2506: &str = "mistral-small-2506";
    pub const PIXTRAL_LARGE_LATEST: &str = "pixtral-large-latest";
    pub const MISTRAL_SMALL_LATEST: &str = "mistral-small-latest";
    pub const MISTRAL_SMALL_2603: &str = "mistral-small-2603";
    pub const MAGISTRAL_MEDIUM_LATEST: &str = "magistral-medium-latest";
    pub const MAGISTRAL_SMALL_LATEST: &str = "magistral-small-latest";
    pub const MAGISTRAL_MEDIUM_2509: &str = "magistral-medium-2509";
    pub const MAGISTRAL_SMALL_2509: &str = "magistral-small-2509";
}

/// Mistral embedding-model constants.
pub mod embedding {
    pub const MISTRAL_EMBED: &str = "mistral-embed";
}

pub const CHAT: &str = chat::MISTRAL_LARGE_LATEST;
pub const EMBEDDING: &str = embedding::MISTRAL_EMBED;

pub const ALL_CHAT: &[&str] = &[
    chat::MINISTRAL_3B_LATEST,
    chat::MINISTRAL_8B_LATEST,
    chat::MINISTRAL_14B_LATEST,
    chat::MISTRAL_LARGE_LATEST,
    chat::MISTRAL_MEDIUM_LATEST,
    chat::MISTRAL_LARGE_2512,
    chat::MISTRAL_MEDIUM_2508,
    chat::MISTRAL_MEDIUM_2505,
    chat::MISTRAL_SMALL_2506,
    chat::PIXTRAL_LARGE_LATEST,
    chat::MISTRAL_SMALL_LATEST,
    chat::MISTRAL_SMALL_2603,
    chat::MAGISTRAL_MEDIUM_LATEST,
    chat::MAGISTRAL_SMALL_LATEST,
    chat::MAGISTRAL_MEDIUM_2509,
    chat::MAGISTRAL_SMALL_2509,
];

pub const ALL_EMBEDDING: &[&str] = &[embedding::MISTRAL_EMBED];
