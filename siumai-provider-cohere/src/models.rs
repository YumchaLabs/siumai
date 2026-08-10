//! Dated Cohere model hints. Model identifiers remain open input.

pub mod embedding {
    pub const EMBED_V4: &str = "embed-v4.0";
    pub const EMBED_V4_FAST: &str = "embed-v4.0-fast";
    pub const EMBED_V4_2B: &str = "embed-v4.0-2b";
    pub const EMBED_V4_FAST_2B: &str = "embed-v4.0-fast-2b";
    pub const EMBED_ENGLISH_V3: &str = "embed-english-v3.0";
    pub const EMBED_ENGLISH_LIGHT_V3: &str = "embed-english-light-v3.0";
    pub const EMBED_MULTILINGUAL_V3: &str = "embed-multilingual-v3.0";
    pub const EMBED_MULTILINGUAL_LIGHT_V3: &str = "embed-multilingual-light-v3.0";

    pub const CURRENT: &[&str] = &[
        EMBED_V4,
        EMBED_V4_FAST,
        EMBED_V4_2B,
        EMBED_V4_FAST_2B,
        EMBED_ENGLISH_V3,
        EMBED_ENGLISH_LIGHT_V3,
        EMBED_MULTILINGUAL_V3,
        EMBED_MULTILINGUAL_LIGHT_V3,
    ];
}

pub mod rerank {
    pub const RERANK_V4_PRO: &str = "rerank-v4.0-pro";
    pub const RERANK_V4_FAST: &str = "rerank-v4.0-fast";
    pub const RERANK_V3_5: &str = "rerank-v3.5";
    pub const RERANK_ENGLISH_V3: &str = "rerank-english-v3.0";
    pub const RERANK_MULTILINGUAL_V3: &str = "rerank-multilingual-v3.0";

    pub const CURRENT: &[&str] = &[
        RERANK_V4_PRO,
        RERANK_V4_FAST,
        RERANK_V3_5,
        RERANK_ENGLISH_V3,
        RERANK_MULTILINGUAL_V3,
    ];
}

pub const CURRENT_EMBEDDING_MODELS: &[&str] = embedding::CURRENT;
pub const CURRENT_RERANK_MODELS: &[&str] = rerank::CURRENT;

pub(crate) fn supports_output_dimension(model: &str) -> bool {
    matches!(
        model,
        embedding::EMBED_V4
            | embedding::EMBED_V4_FAST
            | embedding::EMBED_V4_2B
            | embedding::EMBED_V4_FAST_2B
    )
}
