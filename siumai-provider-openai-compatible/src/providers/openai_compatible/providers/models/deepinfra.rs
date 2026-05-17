//! DeepInfra model constants aligned with the audited AI SDK package subset.
/// DeepInfra chat/language-model constants.
pub mod chat {
    pub const DEEPSEEK_V3: &str = "deepseek-ai/DeepSeek-V3";
    pub const LLAMA_V3P3_70B_INSTRUCT: &str = "meta-llama/Llama-3.3-70B-Instruct";
    pub const LLAMA_V3P1_405B_INSTRUCT: &str = "meta-llama/Meta-Llama-3.1-405B-Instruct";
    pub const LLAMA_V3P1_70B_INSTRUCT: &str = "meta-llama/Meta-Llama-3.1-70B-Instruct";
    pub const LLAMA_V3P1_70B_INSTRUCT_TURBO: &str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo";
    pub const LLAMA_V3P1_8B_INSTRUCT: &str = "meta-llama/Meta-Llama-3.1-8B-Instruct";
    pub const LLAMA_V3P1_8B_INSTRUCT_TURBO: &str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo";
    pub const LLAMA_V3P2_11B_VISION_INSTRUCT: &str = "meta-llama/Llama-3.2-11B-Vision-Instruct";
    pub const LLAMA_V4_MAVERICK_17B_128E_INSTRUCT_FP8: &str =
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8";
    pub const LLAMA_V4_SCOUT_17B_16E_INSTRUCT: &str = "meta-llama/Llama-4-Scout-17B-16E-Instruct";
    pub const MIXTRAL_8X22B_INSTRUCT: &str = "mistralai/Mixtral-8x22B-Instruct-v0.1";
    pub const MIXTRAL_8X7B_INSTRUCT: &str = "mistralai/Mixtral-8x7B-Instruct-v0.1";
    pub const MINICPM_LLAMA3_V_2_5: &str = "openbmb/MiniCPM-Llama3-V-2_5";
    pub const QWEN2P5_72B_INSTRUCT: &str = "Qwen/Qwen2.5-72B-Instruct";
    pub const QWEN2P5_CODER_32B_INSTRUCT: &str = "Qwen/Qwen2.5-Coder-32B-Instruct";
}

/// DeepInfra completion-model constants.
pub mod completion {
    pub const DEEPSEEK_V3: &str = super::chat::DEEPSEEK_V3;
    pub const LLAMA_V3P3_70B_INSTRUCT: &str = super::chat::LLAMA_V3P3_70B_INSTRUCT;
    pub const LLAMA_V3P1_405B_INSTRUCT: &str = super::chat::LLAMA_V3P1_405B_INSTRUCT;
    pub const LLAMA_V3P1_70B_INSTRUCT: &str = super::chat::LLAMA_V3P1_70B_INSTRUCT;
    pub const LLAMA_V3P1_70B_INSTRUCT_TURBO: &str = super::chat::LLAMA_V3P1_70B_INSTRUCT_TURBO;
    pub const LLAMA_V3P1_8B_INSTRUCT: &str = super::chat::LLAMA_V3P1_8B_INSTRUCT;
    pub const LLAMA_V3P1_8B_INSTRUCT_TURBO: &str = super::chat::LLAMA_V3P1_8B_INSTRUCT_TURBO;
    pub const LLAMA_V4_MAVERICK_17B_128E_INSTRUCT_FP8: &str =
        super::chat::LLAMA_V4_MAVERICK_17B_128E_INSTRUCT_FP8;
    pub const LLAMA_V4_SCOUT_17B_16E_INSTRUCT: &str = super::chat::LLAMA_V4_SCOUT_17B_16E_INSTRUCT;
    pub const MIXTRAL_8X22B_INSTRUCT: &str = super::chat::MIXTRAL_8X22B_INSTRUCT;
    pub const MIXTRAL_8X7B_INSTRUCT: &str = super::chat::MIXTRAL_8X7B_INSTRUCT;
    pub const QWEN2P5_72B_INSTRUCT: &str = super::chat::QWEN2P5_72B_INSTRUCT;
    pub const QWEN2P5_CODER_32B_INSTRUCT: &str = super::chat::QWEN2P5_CODER_32B_INSTRUCT;
}

/// DeepInfra embedding-model constants.
pub mod embedding {
    pub const BGE_BASE_EN_V1_5: &str = "BAAI/bge-base-en-v1.5";
    pub const BGE_LARGE_EN_V1_5: &str = "BAAI/bge-large-en-v1.5";
    pub const BGE_M3: &str = "BAAI/bge-m3";
    pub const E5_BASE_V2: &str = "intfloat/e5-base-v2";
    pub const E5_LARGE_V2: &str = "intfloat/e5-large-v2";
    pub const MULTILINGUAL_E5_LARGE: &str = "intfloat/multilingual-e5-large";
    pub const ALL_MINILM_L12_V2: &str = "sentence-transformers/all-MiniLM-L12-v2";
    pub const ALL_MINILM_L6_V2: &str = "sentence-transformers/all-MiniLM-L6-v2";
    pub const ALL_MPNET_BASE_V2: &str = "sentence-transformers/all-mpnet-base-v2";
    pub const CLIP_VIT_B_32: &str = "sentence-transformers/clip-ViT-B-32";
    pub const CLIP_VIT_B_32_MULTILINGUAL_V1: &str =
        "sentence-transformers/clip-ViT-B-32-multilingual-v1";
    pub const MULTI_QA_MPNET_BASE_DOT_V1: &str = "sentence-transformers/multi-qa-mpnet-base-dot-v1";
    pub const PARAPHRASE_MINILM_L6_V2: &str = "sentence-transformers/paraphrase-MiniLM-L6-v2";
    pub const TEXT2VEC_BASE_CHINESE: &str = "shibing624/text2vec-base-chinese";
    pub const GTE_BASE: &str = "thenlper/gte-base";
    pub const GTE_LARGE: &str = "thenlper/gte-large";
}

/// DeepInfra image-model constants.
pub mod image {
    pub const SD3_5: &str = "stabilityai/sd3.5";
    pub const FLUX_1_1_PRO: &str = "black-forest-labs/FLUX-1.1-pro";
    pub const FLUX_1_SCHNELL: &str = "black-forest-labs/FLUX-1-schnell";
    pub const FLUX_1_DEV: &str = "black-forest-labs/FLUX-1-dev";
    pub const FLUX_PRO: &str = "black-forest-labs/FLUX-pro";
    pub const FLUX_1_KONTEXT_DEV: &str = "black-forest-labs/FLUX.1-Kontext-dev";
    pub const FLUX_1_KONTEXT_PRO: &str = "black-forest-labs/FLUX.1-Kontext-pro";
    pub const SD3_5_MEDIUM: &str = "stabilityai/sd3.5-medium";
    pub const SDXL_TURBO: &str = "stabilityai/sdxl-turbo";
}

pub const CHAT: &str = chat::LLAMA_V3P3_70B_INSTRUCT;
pub const COMPLETION: &str = completion::LLAMA_V3P3_70B_INSTRUCT;
pub const EMBEDDING: &str = embedding::BGE_BASE_EN_V1_5;
pub const IMAGE: &str = image::FLUX_1_SCHNELL;

pub const ALL_CHAT: &[&str] = &[
    chat::DEEPSEEK_V3,
    chat::LLAMA_V3P3_70B_INSTRUCT,
    chat::LLAMA_V3P1_405B_INSTRUCT,
    chat::LLAMA_V3P1_70B_INSTRUCT,
    chat::LLAMA_V3P1_70B_INSTRUCT_TURBO,
    chat::LLAMA_V3P1_8B_INSTRUCT,
    chat::LLAMA_V3P1_8B_INSTRUCT_TURBO,
    chat::LLAMA_V3P2_11B_VISION_INSTRUCT,
    chat::LLAMA_V4_MAVERICK_17B_128E_INSTRUCT_FP8,
    chat::LLAMA_V4_SCOUT_17B_16E_INSTRUCT,
    chat::MIXTRAL_8X22B_INSTRUCT,
    chat::MIXTRAL_8X7B_INSTRUCT,
    chat::MINICPM_LLAMA3_V_2_5,
    chat::QWEN2P5_72B_INSTRUCT,
    chat::QWEN2P5_CODER_32B_INSTRUCT,
];

pub const ALL_COMPLETION: &[&str] = &[
    completion::DEEPSEEK_V3,
    completion::LLAMA_V3P3_70B_INSTRUCT,
    completion::LLAMA_V3P1_405B_INSTRUCT,
    completion::LLAMA_V3P1_70B_INSTRUCT,
    completion::LLAMA_V3P1_70B_INSTRUCT_TURBO,
    completion::LLAMA_V3P1_8B_INSTRUCT,
    completion::LLAMA_V3P1_8B_INSTRUCT_TURBO,
    completion::LLAMA_V4_MAVERICK_17B_128E_INSTRUCT_FP8,
    completion::LLAMA_V4_SCOUT_17B_16E_INSTRUCT,
    completion::MIXTRAL_8X22B_INSTRUCT,
    completion::MIXTRAL_8X7B_INSTRUCT,
    completion::QWEN2P5_72B_INSTRUCT,
    completion::QWEN2P5_CODER_32B_INSTRUCT,
];

pub const ALL_EMBEDDING: &[&str] = &[
    embedding::BGE_BASE_EN_V1_5,
    embedding::BGE_LARGE_EN_V1_5,
    embedding::BGE_M3,
    embedding::E5_BASE_V2,
    embedding::E5_LARGE_V2,
    embedding::MULTILINGUAL_E5_LARGE,
    embedding::ALL_MINILM_L12_V2,
    embedding::ALL_MINILM_L6_V2,
    embedding::ALL_MPNET_BASE_V2,
    embedding::CLIP_VIT_B_32,
    embedding::CLIP_VIT_B_32_MULTILINGUAL_V1,
    embedding::MULTI_QA_MPNET_BASE_DOT_V1,
    embedding::PARAPHRASE_MINILM_L6_V2,
    embedding::TEXT2VEC_BASE_CHINESE,
    embedding::GTE_BASE,
    embedding::GTE_LARGE,
];

pub const ALL_IMAGE: &[&str] = &[
    image::SD3_5,
    image::FLUX_1_1_PRO,
    image::FLUX_1_SCHNELL,
    image::FLUX_1_DEV,
    image::FLUX_PRO,
    image::FLUX_1_KONTEXT_DEV,
    image::FLUX_1_KONTEXT_PRO,
    image::SD3_5_MEDIUM,
    image::SDXL_TURBO,
];

/// Get all curated DeepInfra models from the audited AI SDK subset.
pub fn all_models() -> Vec<String> {
    let mut models = Vec::new();

    for model in ALL_CHAT
        .iter()
        .chain(ALL_COMPLETION.iter())
        .chain(ALL_EMBEDDING.iter())
        .chain(ALL_IMAGE.iter())
    {
        if !models.iter().any(|existing| existing == model) {
            models.push((*model).to_string());
        }
    }

    models
}
