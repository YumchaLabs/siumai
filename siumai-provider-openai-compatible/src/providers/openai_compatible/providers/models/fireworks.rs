//! Fireworks model constants aligned with the audited AI SDK package subset.
/// Fireworks chat/language-model constants.
pub mod chat {
    pub const DEEPSEEK_V3: &str = "accounts/fireworks/models/deepseek-v3";
    pub const LLAMA_V3P3_70B_INSTRUCT: &str = "accounts/fireworks/models/llama-v3p3-70b-instruct";
    pub const LLAMA_V3P2_3B_INSTRUCT: &str = "accounts/fireworks/models/llama-v3p2-3b-instruct";
    pub const LLAMA_V3P1_405B_INSTRUCT: &str = "accounts/fireworks/models/llama-v3p1-405b-instruct";
    pub const LLAMA_V3P1_8B_INSTRUCT: &str = "accounts/fireworks/models/llama-v3p1-8b-instruct";
    pub const MIXTRAL_8X7B_INSTRUCT: &str = "accounts/fireworks/models/mixtral-8x7b-instruct";
    pub const MIXTRAL_8X22B_INSTRUCT: &str = "accounts/fireworks/models/mixtral-8x22b-instruct";
    pub const MIXTRAL_8X7B_INSTRUCT_HF: &str = "accounts/fireworks/models/mixtral-8x7b-instruct-hf";
    pub const QWEN2P5_CODER_32B_INSTRUCT: &str =
        "accounts/fireworks/models/qwen2p5-coder-32b-instruct";
    pub const QWEN2P5_72B_INSTRUCT: &str = "accounts/fireworks/models/qwen2p5-72b-instruct";
    pub const QWEN_QWQ_32B_PREVIEW: &str = "accounts/fireworks/models/qwen-qwq-32b-preview";
    pub const QWEN2_VL_72B_INSTRUCT: &str = "accounts/fireworks/models/qwen2-vl-72b-instruct";
    pub const LLAMA_V3P2_11B_VISION_INSTRUCT: &str =
        "accounts/fireworks/models/llama-v3p2-11b-vision-instruct";
    pub const QWQ_32B: &str = "accounts/fireworks/models/qwq-32b";
    pub const YI_LARGE: &str = "accounts/fireworks/models/yi-large";
    pub const KIMI_K2_INSTRUCT: &str = "accounts/fireworks/models/kimi-k2-instruct";
    pub const KIMI_K2_THINKING: &str = "accounts/fireworks/models/kimi-k2-thinking";
    pub const KIMI_K2P5: &str = "accounts/fireworks/models/kimi-k2p5";
    pub const MINIMAX_M2: &str = "accounts/fireworks/models/minimax-m2";
}

/// Fireworks completion-model constants.
pub mod completion {
    pub const LLAMA_V3_8B_INSTRUCT: &str = "accounts/fireworks/models/llama-v3-8b-instruct";
    pub const LLAMA_V2_34B_CODE: &str = "accounts/fireworks/models/llama-v2-34b-code";
}

/// Fireworks embedding-model constants.
pub mod embedding {
    pub const NOMIC_EMBED_TEXT_V1_5: &str = "nomic-ai/nomic-embed-text-v1.5";
}

/// Fireworks image-model constants.
pub mod image {
    pub const FLUX_1_DEV_FP8: &str = "accounts/fireworks/models/flux-1-dev-fp8";
    pub const FLUX_1_SCHNELL_FP8: &str = "accounts/fireworks/models/flux-1-schnell-fp8";
    pub const FLUX_KONTEXT_PRO: &str = "accounts/fireworks/models/flux-kontext-pro";
    pub const FLUX_KONTEXT_MAX: &str = "accounts/fireworks/models/flux-kontext-max";
    pub const PLAYGROUND_V2_5_1024PX_AESTHETIC: &str =
        "accounts/fireworks/models/playground-v2-5-1024px-aesthetic";
    pub const JAPANESE_STABLE_DIFFUSION_XL: &str =
        "accounts/fireworks/models/japanese-stable-diffusion-xl";
    pub const PLAYGROUND_V2_1024PX_AESTHETIC: &str =
        "accounts/fireworks/models/playground-v2-1024px-aesthetic";
    pub const SSD_1B: &str = "accounts/fireworks/models/SSD-1B";
    pub const STABLE_DIFFUSION_XL_1024_V1_0: &str =
        "accounts/fireworks/models/stable-diffusion-xl-1024-v1-0";
}

pub const CHAT: &str = chat::LLAMA_V3P1_8B_INSTRUCT;
pub const COMPLETION: &str = completion::LLAMA_V3_8B_INSTRUCT;
pub const EMBEDDING: &str = embedding::NOMIC_EMBED_TEXT_V1_5;
pub const IMAGE: &str = image::FLUX_1_DEV_FP8;

pub const ALL_CHAT: &[&str] = &[
    chat::DEEPSEEK_V3,
    chat::LLAMA_V3P3_70B_INSTRUCT,
    chat::LLAMA_V3P2_3B_INSTRUCT,
    chat::LLAMA_V3P1_405B_INSTRUCT,
    chat::LLAMA_V3P1_8B_INSTRUCT,
    chat::MIXTRAL_8X7B_INSTRUCT,
    chat::MIXTRAL_8X22B_INSTRUCT,
    chat::MIXTRAL_8X7B_INSTRUCT_HF,
    chat::QWEN2P5_CODER_32B_INSTRUCT,
    chat::QWEN2P5_72B_INSTRUCT,
    chat::QWEN_QWQ_32B_PREVIEW,
    chat::QWEN2_VL_72B_INSTRUCT,
    chat::LLAMA_V3P2_11B_VISION_INSTRUCT,
    chat::QWQ_32B,
    chat::YI_LARGE,
    chat::KIMI_K2_INSTRUCT,
    chat::KIMI_K2_THINKING,
    chat::KIMI_K2P5,
    chat::MINIMAX_M2,
];

pub const ALL_COMPLETION: &[&str] = &[
    completion::LLAMA_V3_8B_INSTRUCT,
    completion::LLAMA_V2_34B_CODE,
];

pub const ALL_EMBEDDING: &[&str] = &[embedding::NOMIC_EMBED_TEXT_V1_5];

pub const ALL_IMAGE: &[&str] = &[
    image::FLUX_1_DEV_FP8,
    image::FLUX_1_SCHNELL_FP8,
    image::FLUX_KONTEXT_PRO,
    image::FLUX_KONTEXT_MAX,
    image::PLAYGROUND_V2_5_1024PX_AESTHETIC,
    image::JAPANESE_STABLE_DIFFUSION_XL,
    image::PLAYGROUND_V2_1024PX_AESTHETIC,
    image::SSD_1B,
    image::STABLE_DIFFUSION_XL_1024_V1_0,
];

/// Get all curated Fireworks models from the audited AI SDK subset.
pub fn all_models() -> Vec<String> {
    let mut models = Vec::new();
    models.extend(ALL_CHAT.iter().map(|&s| s.to_string()));
    models.extend(ALL_COMPLETION.iter().map(|&s| s.to_string()));
    models.extend(ALL_EMBEDDING.iter().map(|&s| s.to_string()));
    models.extend(ALL_IMAGE.iter().map(|&s| s.to_string()));
    models
}
