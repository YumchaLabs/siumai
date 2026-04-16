//! OpenAI-Compatible Provider Model Definitions
//!
//! This module contains model definitions for various OpenAI-compatible providers.

/// `DeepSeek` model constants
pub mod deepseek {
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
}

/// `OpenRouter` model constants
pub mod openrouter {
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
}

/// DeepInfra model constants aligned with the audited AI SDK package subset.
pub mod deepinfra {
    /// DeepInfra chat/language-model constants.
    pub mod chat {
        pub const DEEPSEEK_V3: &str = "deepseek-ai/DeepSeek-V3";
        pub const LLAMA_V3P3_70B_INSTRUCT: &str = "meta-llama/Llama-3.3-70B-Instruct";
        pub const LLAMA_V3P1_405B_INSTRUCT: &str = "meta-llama/Meta-Llama-3.1-405B-Instruct";
        pub const LLAMA_V3P1_70B_INSTRUCT: &str = "meta-llama/Meta-Llama-3.1-70B-Instruct";
        pub const LLAMA_V3P1_70B_INSTRUCT_TURBO: &str =
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo";
        pub const LLAMA_V3P1_8B_INSTRUCT: &str = "meta-llama/Meta-Llama-3.1-8B-Instruct";
        pub const LLAMA_V3P1_8B_INSTRUCT_TURBO: &str =
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo";
        pub const LLAMA_V3P2_11B_VISION_INSTRUCT: &str = "meta-llama/Llama-3.2-11B-Vision-Instruct";
        pub const LLAMA_V4_MAVERICK_17B_128E_INSTRUCT_FP8: &str =
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8";
        pub const LLAMA_V4_SCOUT_17B_16E_INSTRUCT: &str =
            "meta-llama/Llama-4-Scout-17B-16E-Instruct";
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
        pub const LLAMA_V4_SCOUT_17B_16E_INSTRUCT: &str =
            super::chat::LLAMA_V4_SCOUT_17B_16E_INSTRUCT;
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
        pub const MULTI_QA_MPNET_BASE_DOT_V1: &str =
            "sentence-transformers/multi-qa-mpnet-base-dot-v1";
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
}

/// Vertex MaaS model constants aligned with the audited AI SDK package subset.
pub mod vertex_maas {
    /// Vertex MaaS chat/language-model constants.
    pub mod chat {
        pub const DEEPSEEK_R1_0528_MAAS: &str = "deepseek-ai/deepseek-r1-0528-maas";
        pub const DEEPSEEK_V3_1_MAAS: &str = "deepseek-ai/deepseek-v3.1-maas";
        pub const DEEPSEEK_V3_2_MAAS: &str = "deepseek-ai/deepseek-v3.2-maas";
        pub const GPT_OSS_120B_MAAS: &str = "openai/gpt-oss-120b-maas";
        pub const GPT_OSS_20B_MAAS: &str = "openai/gpt-oss-20b-maas";
        pub const LLAMA_4_MAVERICK_17B_128E_INSTRUCT_MAAS: &str =
            "meta/llama-4-maverick-17b-128e-instruct-maas";
        pub const LLAMA_4_SCOUT_17B_16E_INSTRUCT_MAAS: &str =
            "meta/llama-4-scout-17b-16e-instruct-maas";
        pub const MINIMAX_M2_MAAS: &str = "minimax/minimax-m2-maas";
        pub const QWEN3_CODER_480B_A35B_INSTRUCT_MAAS: &str =
            "qwen/qwen3-coder-480b-a35b-instruct-maas";
        pub const QWEN3_NEXT_80B_A3B_INSTRUCT_MAAS: &str = "qwen/qwen3-next-80b-a3b-instruct-maas";
        pub const QWEN3_NEXT_80B_A3B_THINKING_MAAS: &str = "qwen/qwen3-next-80b-a3b-thinking-maas";
        pub const KIMI_K2_THINKING_MAAS: &str = "moonshotai/kimi-k2-thinking-maas";
    }

    /// Vertex MaaS completion-model constants.
    pub mod completion {
        pub const DEEPSEEK_R1_0528_MAAS: &str = super::chat::DEEPSEEK_R1_0528_MAAS;
        pub const DEEPSEEK_V3_1_MAAS: &str = super::chat::DEEPSEEK_V3_1_MAAS;
        pub const DEEPSEEK_V3_2_MAAS: &str = super::chat::DEEPSEEK_V3_2_MAAS;
        pub const GPT_OSS_120B_MAAS: &str = super::chat::GPT_OSS_120B_MAAS;
        pub const GPT_OSS_20B_MAAS: &str = super::chat::GPT_OSS_20B_MAAS;
        pub const LLAMA_4_MAVERICK_17B_128E_INSTRUCT_MAAS: &str =
            super::chat::LLAMA_4_MAVERICK_17B_128E_INSTRUCT_MAAS;
        pub const LLAMA_4_SCOUT_17B_16E_INSTRUCT_MAAS: &str =
            super::chat::LLAMA_4_SCOUT_17B_16E_INSTRUCT_MAAS;
        pub const MINIMAX_M2_MAAS: &str = super::chat::MINIMAX_M2_MAAS;
        pub const QWEN3_CODER_480B_A35B_INSTRUCT_MAAS: &str =
            super::chat::QWEN3_CODER_480B_A35B_INSTRUCT_MAAS;
        pub const QWEN3_NEXT_80B_A3B_INSTRUCT_MAAS: &str =
            super::chat::QWEN3_NEXT_80B_A3B_INSTRUCT_MAAS;
        pub const QWEN3_NEXT_80B_A3B_THINKING_MAAS: &str =
            super::chat::QWEN3_NEXT_80B_A3B_THINKING_MAAS;
        pub const KIMI_K2_THINKING_MAAS: &str = super::chat::KIMI_K2_THINKING_MAAS;
    }

    /// Vertex MaaS embedding-model constants.
    pub mod embedding {
        pub const DEEPSEEK_R1_0528_MAAS: &str = super::chat::DEEPSEEK_R1_0528_MAAS;
        pub const DEEPSEEK_V3_1_MAAS: &str = super::chat::DEEPSEEK_V3_1_MAAS;
        pub const DEEPSEEK_V3_2_MAAS: &str = super::chat::DEEPSEEK_V3_2_MAAS;
        pub const GPT_OSS_120B_MAAS: &str = super::chat::GPT_OSS_120B_MAAS;
        pub const GPT_OSS_20B_MAAS: &str = super::chat::GPT_OSS_20B_MAAS;
        pub const LLAMA_4_MAVERICK_17B_128E_INSTRUCT_MAAS: &str =
            super::chat::LLAMA_4_MAVERICK_17B_128E_INSTRUCT_MAAS;
        pub const LLAMA_4_SCOUT_17B_16E_INSTRUCT_MAAS: &str =
            super::chat::LLAMA_4_SCOUT_17B_16E_INSTRUCT_MAAS;
        pub const MINIMAX_M2_MAAS: &str = super::chat::MINIMAX_M2_MAAS;
        pub const QWEN3_CODER_480B_A35B_INSTRUCT_MAAS: &str =
            super::chat::QWEN3_CODER_480B_A35B_INSTRUCT_MAAS;
        pub const QWEN3_NEXT_80B_A3B_INSTRUCT_MAAS: &str =
            super::chat::QWEN3_NEXT_80B_A3B_INSTRUCT_MAAS;
        pub const QWEN3_NEXT_80B_A3B_THINKING_MAAS: &str =
            super::chat::QWEN3_NEXT_80B_A3B_THINKING_MAAS;
        pub const KIMI_K2_THINKING_MAAS: &str = super::chat::KIMI_K2_THINKING_MAAS;
    }

    pub const CHAT: &str = chat::DEEPSEEK_V3_2_MAAS;
    pub const COMPLETION: &str = completion::DEEPSEEK_V3_2_MAAS;
    pub const EMBEDDING: &str = embedding::DEEPSEEK_V3_2_MAAS;

    pub const ALL_CHAT: &[&str] = &[
        chat::DEEPSEEK_R1_0528_MAAS,
        chat::DEEPSEEK_V3_1_MAAS,
        chat::DEEPSEEK_V3_2_MAAS,
        chat::GPT_OSS_120B_MAAS,
        chat::GPT_OSS_20B_MAAS,
        chat::LLAMA_4_MAVERICK_17B_128E_INSTRUCT_MAAS,
        chat::LLAMA_4_SCOUT_17B_16E_INSTRUCT_MAAS,
        chat::MINIMAX_M2_MAAS,
        chat::QWEN3_CODER_480B_A35B_INSTRUCT_MAAS,
        chat::QWEN3_NEXT_80B_A3B_INSTRUCT_MAAS,
        chat::QWEN3_NEXT_80B_A3B_THINKING_MAAS,
        chat::KIMI_K2_THINKING_MAAS,
    ];

    pub const ALL_COMPLETION: &[&str] = &[
        completion::DEEPSEEK_R1_0528_MAAS,
        completion::DEEPSEEK_V3_1_MAAS,
        completion::DEEPSEEK_V3_2_MAAS,
        completion::GPT_OSS_120B_MAAS,
        completion::GPT_OSS_20B_MAAS,
        completion::LLAMA_4_MAVERICK_17B_128E_INSTRUCT_MAAS,
        completion::LLAMA_4_SCOUT_17B_16E_INSTRUCT_MAAS,
        completion::MINIMAX_M2_MAAS,
        completion::QWEN3_CODER_480B_A35B_INSTRUCT_MAAS,
        completion::QWEN3_NEXT_80B_A3B_INSTRUCT_MAAS,
        completion::QWEN3_NEXT_80B_A3B_THINKING_MAAS,
        completion::KIMI_K2_THINKING_MAAS,
    ];

    pub const ALL_EMBEDDING: &[&str] = &[
        embedding::DEEPSEEK_R1_0528_MAAS,
        embedding::DEEPSEEK_V3_1_MAAS,
        embedding::DEEPSEEK_V3_2_MAAS,
        embedding::GPT_OSS_120B_MAAS,
        embedding::GPT_OSS_20B_MAAS,
        embedding::LLAMA_4_MAVERICK_17B_128E_INSTRUCT_MAAS,
        embedding::LLAMA_4_SCOUT_17B_16E_INSTRUCT_MAAS,
        embedding::MINIMAX_M2_MAAS,
        embedding::QWEN3_CODER_480B_A35B_INSTRUCT_MAAS,
        embedding::QWEN3_NEXT_80B_A3B_INSTRUCT_MAAS,
        embedding::QWEN3_NEXT_80B_A3B_THINKING_MAAS,
        embedding::KIMI_K2_THINKING_MAAS,
    ];

    /// Get all curated Vertex MaaS models from the audited AI SDK subset.
    pub fn all_models() -> Vec<String> {
        let mut models = Vec::new();

        for model in ALL_CHAT
            .iter()
            .chain(ALL_COMPLETION.iter())
            .chain(ALL_EMBEDDING.iter())
        {
            if !models.iter().any(|existing| existing == model) {
                models.push((*model).to_string());
            }
        }

        models
    }
}

/// Mistral model constants aligned with the audited AI SDK package subset.
pub mod mistral {
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
}

/// Perplexity model constants aligned with the audited AI SDK package subset.
pub mod perplexity {
    /// Perplexity chat/language-model constants.
    pub mod chat {
        pub const SONAR_DEEP_RESEARCH: &str = "sonar-deep-research";
        pub const SONAR_REASONING_PRO: &str = "sonar-reasoning-pro";
        pub const SONAR_REASONING: &str = "sonar-reasoning";
        pub const SONAR_PRO: &str = "sonar-pro";
        pub const SONAR: &str = "sonar";
    }

    pub const CHAT: &str = chat::SONAR;

    pub const ALL_CHAT: &[&str] = &[
        chat::SONAR_DEEP_RESEARCH,
        chat::SONAR_REASONING_PRO,
        chat::SONAR_REASONING,
        chat::SONAR_PRO,
        chat::SONAR,
    ];
}

/// Fireworks model constants aligned with the audited AI SDK package subset.
pub mod fireworks {
    /// Fireworks chat/language-model constants.
    pub mod chat {
        pub const DEEPSEEK_V3: &str = "accounts/fireworks/models/deepseek-v3";
        pub const LLAMA_V3P3_70B_INSTRUCT: &str =
            "accounts/fireworks/models/llama-v3p3-70b-instruct";
        pub const LLAMA_V3P2_3B_INSTRUCT: &str = "accounts/fireworks/models/llama-v3p2-3b-instruct";
        pub const LLAMA_V3P1_405B_INSTRUCT: &str =
            "accounts/fireworks/models/llama-v3p1-405b-instruct";
        pub const LLAMA_V3P1_8B_INSTRUCT: &str = "accounts/fireworks/models/llama-v3p1-8b-instruct";
        pub const MIXTRAL_8X7B_INSTRUCT: &str = "accounts/fireworks/models/mixtral-8x7b-instruct";
        pub const MIXTRAL_8X22B_INSTRUCT: &str = "accounts/fireworks/models/mixtral-8x22b-instruct";
        pub const MIXTRAL_8X7B_INSTRUCT_HF: &str =
            "accounts/fireworks/models/mixtral-8x7b-instruct-hf";
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
}

/// xAI model constants
pub mod xai {
    /// Grok Beta model
    pub const GROK_BETA: &str = "grok-beta";
    /// Grok Vision Beta model
    pub const GROK_VISION_BETA: &str = "grok-vision-beta";

    /// Get all xAI models
    pub fn all_models() -> Vec<String> {
        vec![GROK_BETA.to_string(), GROK_VISION_BETA.to_string()]
    }
}

/// SiliconFlow model constants
///
/// These constants are based on the actual models available from SiliconFlow API.
/// Updated to reflect the current model offerings as of the latest API query.
pub mod siliconflow {
    // ========================================================================
    // Chat Models - Most Popular and Recommended
    // ========================================================================

    /// DeepSeek V3.1 - Latest flagship reasoning model
    pub const DEEPSEEK_V3_1: &str = "deepseek-ai/DeepSeek-V3.1";
    /// DeepSeek V3.1 Pro version
    pub const DEEPSEEK_V3_1_PRO: &str = "Pro/deepseek-ai/DeepSeek-V3.1";
    /// DeepSeek V3 - Previous flagship model
    pub const DEEPSEEK_V3: &str = "deepseek-ai/DeepSeek-V3";
    /// DeepSeek V3 Pro version
    pub const DEEPSEEK_V3_PRO: &str = "Pro/deepseek-ai/DeepSeek-V3";
    /// DeepSeek R1 - Reasoning model
    pub const DEEPSEEK_R1: &str = "deepseek-ai/DeepSeek-R1";
    /// DeepSeek R1 Pro version
    pub const DEEPSEEK_R1_PRO: &str = "Pro/deepseek-ai/DeepSeek-R1";
    /// DeepSeek V2.5 - Previous generation
    pub const DEEPSEEK_V2_5: &str = "deepseek-ai/DeepSeek-V2.5";
    /// DeepSeek VL2 - Vision-language model
    pub const DEEPSEEK_VL2: &str = "deepseek-ai/deepseek-vl2";

    /// Qwen 3 235B A22B - Latest flagship
    pub const QWEN3_235B_A22B: &str = "Qwen/Qwen3-235B-A22B";
    /// Qwen 3 235B A22B Instruct
    pub const QWEN3_235B_A22B_INSTRUCT: &str = "Qwen/Qwen3-235B-A22B-Instruct-2507";
    /// Qwen 3 235B A22B Thinking
    pub const QWEN3_235B_A22B_THINKING: &str = "Qwen/Qwen3-235B-A22B-Thinking-2507";
    /// Qwen 3 32B
    pub const QWEN3_32B: &str = "Qwen/Qwen3-32B";
    /// Qwen 3 30B A3B
    pub const QWEN3_30B_A3B: &str = "Qwen/Qwen3-30B-A3B";
    /// Qwen 3 30B A3B Instruct
    pub const QWEN3_30B_A3B_INSTRUCT: &str = "Qwen/Qwen3-30B-A3B-Instruct-2507";
    /// Qwen 3 30B A3B Thinking
    pub const QWEN3_30B_A3B_THINKING: &str = "Qwen/Qwen3-30B-A3B-Thinking-2507";
    /// Qwen 3 14B
    pub const QWEN3_14B: &str = "Qwen/Qwen3-14B";
    /// Qwen 3 8B
    pub const QWEN3_8B: &str = "Qwen/Qwen3-8B";

    /// Qwen 2.5 72B Instruct
    pub const QWEN_2_5_72B_INSTRUCT: &str = "Qwen/Qwen2.5-72B-Instruct";
    /// Qwen 2.5 72B Instruct 128K context
    pub const QWEN_2_5_72B_INSTRUCT_128K: &str = "Qwen/Qwen2.5-72B-Instruct-128K";
    /// Qwen 2.5 32B Instruct
    pub const QWEN_2_5_32B_INSTRUCT: &str = "Qwen/Qwen2.5-32B-Instruct";
    /// Qwen 2.5 14B Instruct
    pub const QWEN_2_5_14B_INSTRUCT: &str = "Qwen/Qwen2.5-14B-Instruct";
    /// Qwen 2.5 7B Instruct
    pub const QWEN_2_5_7B_INSTRUCT: &str = "Qwen/Qwen2.5-7B-Instruct";

    /// Qwen 2.5 VL 72B Instruct - Vision-language model
    pub const QWEN_2_5_VL_72B_INSTRUCT: &str = "Qwen/Qwen2.5-VL-72B-Instruct";
    /// Qwen 2.5 VL 32B Instruct - Vision-language model
    pub const QWEN_2_5_VL_32B_INSTRUCT: &str = "Qwen/Qwen2.5-VL-32B-Instruct";
    /// Qwen 2.5 VL 7B Instruct Pro - Vision-language model
    pub const QWEN_2_5_VL_7B_INSTRUCT_PRO: &str = "Pro/Qwen/Qwen2.5-VL-7B-Instruct";

    /// Qwen 2.5 Coder models
    pub const QWEN_2_5_CODER_32B_INSTRUCT: &str = "Qwen/Qwen2.5-Coder-32B-Instruct";
    pub const QWEN_2_5_CODER_7B_INSTRUCT: &str = "Qwen/Qwen2.5-Coder-7B-Instruct";
    pub const QWEN_2_5_CODER_7B_INSTRUCT_PRO: &str = "Pro/Qwen/Qwen2.5-Coder-7B-Instruct";

    /// Qwen 3 Coder models
    pub const QWEN3_CODER_480B_A35B_INSTRUCT: &str = "Qwen/Qwen3-Coder-480B-A35B-Instruct";
    pub const QWEN3_CODER_30B_A3B_INSTRUCT: &str = "Qwen/Qwen3-Coder-30B-A3B-Instruct";

    /// QwQ 32B - Reasoning model
    pub const QWQ_32B: &str = "Qwen/QwQ-32B";
    /// QVQ 72B Preview - Vision-question model
    pub const QVQ_72B_PREVIEW: &str = "Qwen/QVQ-72B-Preview";

    /// Kimi K2 Instruct
    pub const KIMI_K2_INSTRUCT: &str = "moonshotai/Kimi-K2-Instruct";
    /// Kimi K2 Instruct Pro
    pub const KIMI_K2_INSTRUCT_PRO: &str = "Pro/moonshotai/Kimi-K2-Instruct";

    /// GLM models
    pub const GLM_4_5: &str = "zai-org/GLM-4.5";
    pub const GLM_4_5_AIR: &str = "zai-org/GLM-4.5-Air";
    pub const GLM_4_5V: &str = "zai-org/GLM-4.5V";
    pub const GLM_4_1V_9B_THINKING: &str = "THUDM/GLM-4.1V-9B-Thinking";
    pub const GLM_4_1V_9B_THINKING_PRO: &str = "Pro/THUDM/GLM-4.1V-9B-Thinking";

    // ========================================================================
    // Embedding Models
    // ========================================================================

    /// BCE Embedding Base V1 - NetEase Youdao
    pub const BCE_EMBEDDING_BASE_V1: &str = "netease-youdao/bce-embedding-base_v1";

    /// Qwen 3 Embedding models
    pub const QWEN3_EMBEDDING_8B: &str = "Qwen/Qwen3-Embedding-8B";
    pub const QWEN3_EMBEDDING_4B: &str = "Qwen/Qwen3-Embedding-4B";
    pub const QWEN3_EMBEDDING_0_6B: &str = "Qwen/Qwen3-Embedding-0.6B";

    /// BGE models (legacy, still available)
    pub const BGE_LARGE_EN_V1_5: &str = "BAAI/bge-large-en-v1.5";
    pub const BGE_LARGE_ZH_V1_5: &str = "BAAI/bge-large-zh-v1.5";
    pub const BGE_M3: &str = "BAAI/bge-m3";
    pub const BGE_M3_PRO: &str = "Pro/BAAI/bge-m3";

    // ========================================================================
    // Rerank Models
    // ========================================================================

    /// BGE Reranker V2 M3
    pub const BGE_RERANKER_V2_M3: &str = "BAAI/bge-reranker-v2-m3";
    /// BGE Reranker V2 M3 Pro
    pub const BGE_RERANKER_V2_M3_PRO: &str = "Pro/BAAI/bge-reranker-v2-m3";

    /// BCE Reranker Base V1 - NetEase Youdao
    pub const BCE_RERANKER_BASE_V1: &str = "netease-youdao/bce-reranker-base_v1";

    /// Qwen 3 Reranker models
    pub const QWEN3_RERANKER_8B: &str = "Qwen/Qwen3-Reranker-8B";
    pub const QWEN3_RERANKER_4B: &str = "Qwen/Qwen3-Reranker-4B";
    pub const QWEN3_RERANKER_0_6B: &str = "Qwen/Qwen3-Reranker-0.6B";

    // ========================================================================
    // Image Generation Models
    // ========================================================================

    /// FLUX.1 models
    pub const FLUX_1_PRO: &str = "black-forest-labs/FLUX.1-pro";
    pub const FLUX_1_DEV: &str = "black-forest-labs/FLUX.1-dev";
    pub const FLUX_1_SCHNELL: &str = "black-forest-labs/FLUX.1-schnell";
    pub const FLUX_1_SCHNELL_PRO: &str = "Pro/black-forest-labs/FLUX.1-schnell";
    pub const FLUX_1_DEV_LORA: &str = "LoRA/black-forest-labs/FLUX.1-dev";

    /// Stable Diffusion models
    pub const STABLE_DIFFUSION_3_5_LARGE: &str = "stabilityai/stable-diffusion-3.5-large";
    pub const STABLE_DIFFUSION_XL_BASE_1_0: &str = "stabilityai/stable-diffusion-xl-base-1.0";

    /// Kolors
    pub const KOLORS: &str = "Kwai-Kolors/Kolors";

    /// Get popular SiliconFlow chat models (most commonly used)
    pub fn popular_chat_models() -> Vec<String> {
        vec![
            DEEPSEEK_V3_1.to_string(),
            DEEPSEEK_V3.to_string(),
            DEEPSEEK_R1.to_string(),
            QWEN3_235B_A22B_INSTRUCT.to_string(),
            QWEN3_30B_A3B_INSTRUCT.to_string(),
            QWEN_2_5_72B_INSTRUCT.to_string(),
            QWEN_2_5_32B_INSTRUCT.to_string(),
            QWEN_2_5_14B_INSTRUCT.to_string(),
            QWEN_2_5_7B_INSTRUCT.to_string(),
            KIMI_K2_INSTRUCT.to_string(),
            GLM_4_5.to_string(),
        ]
    }

    /// Get all SiliconFlow chat models
    pub fn all_chat_models() -> Vec<String> {
        vec![
            // DeepSeek models
            DEEPSEEK_V3_1.to_string(),
            DEEPSEEK_V3_1_PRO.to_string(),
            DEEPSEEK_V3.to_string(),
            DEEPSEEK_V3_PRO.to_string(),
            DEEPSEEK_R1.to_string(),
            DEEPSEEK_R1_PRO.to_string(),
            DEEPSEEK_V2_5.to_string(),
            DEEPSEEK_VL2.to_string(),
            // Qwen 3 models
            QWEN3_235B_A22B.to_string(),
            QWEN3_235B_A22B_INSTRUCT.to_string(),
            QWEN3_235B_A22B_THINKING.to_string(),
            QWEN3_32B.to_string(),
            QWEN3_30B_A3B.to_string(),
            QWEN3_30B_A3B_INSTRUCT.to_string(),
            QWEN3_30B_A3B_THINKING.to_string(),
            QWEN3_14B.to_string(),
            QWEN3_8B.to_string(),
            // Qwen 2.5 models
            QWEN_2_5_72B_INSTRUCT.to_string(),
            QWEN_2_5_72B_INSTRUCT_128K.to_string(),
            QWEN_2_5_32B_INSTRUCT.to_string(),
            QWEN_2_5_14B_INSTRUCT.to_string(),
            QWEN_2_5_7B_INSTRUCT.to_string(),
            // Vision-language models
            QWEN_2_5_VL_72B_INSTRUCT.to_string(),
            QWEN_2_5_VL_32B_INSTRUCT.to_string(),
            QWEN_2_5_VL_7B_INSTRUCT_PRO.to_string(),
            // Coder models
            QWEN_2_5_CODER_32B_INSTRUCT.to_string(),
            QWEN_2_5_CODER_7B_INSTRUCT.to_string(),
            QWEN_2_5_CODER_7B_INSTRUCT_PRO.to_string(),
            QWEN3_CODER_480B_A35B_INSTRUCT.to_string(),
            QWEN3_CODER_30B_A3B_INSTRUCT.to_string(),
            // Reasoning models
            QWQ_32B.to_string(),
            QVQ_72B_PREVIEW.to_string(),
            // Kimi models
            KIMI_K2_INSTRUCT.to_string(),
            KIMI_K2_INSTRUCT_PRO.to_string(),
            // GLM models
            GLM_4_5.to_string(),
            GLM_4_5_AIR.to_string(),
            GLM_4_5V.to_string(),
            GLM_4_1V_9B_THINKING.to_string(),
            GLM_4_1V_9B_THINKING_PRO.to_string(),
        ]
    }

    /// Get all SiliconFlow embedding models
    pub fn all_embedding_models() -> Vec<String> {
        vec![
            // Recommended embedding models
            BCE_EMBEDDING_BASE_V1.to_string(),
            QWEN3_EMBEDDING_8B.to_string(),
            QWEN3_EMBEDDING_4B.to_string(),
            QWEN3_EMBEDDING_0_6B.to_string(),
            // BGE models (legacy but still available)
            BGE_LARGE_EN_V1_5.to_string(),
            BGE_LARGE_ZH_V1_5.to_string(),
            BGE_M3.to_string(),
            BGE_M3_PRO.to_string(),
        ]
    }

    /// Get all SiliconFlow rerank models
    pub fn all_rerank_models() -> Vec<String> {
        vec![
            BGE_RERANKER_V2_M3.to_string(),
            BGE_RERANKER_V2_M3_PRO.to_string(),
            QWEN3_RERANKER_8B.to_string(),
            QWEN3_RERANKER_4B.to_string(),
            QWEN3_RERANKER_0_6B.to_string(),
            BCE_RERANKER_BASE_V1.to_string(),
        ]
    }

    /// Get all SiliconFlow image generation models
    pub fn all_image_models() -> Vec<String> {
        vec![
            // FLUX models
            FLUX_1_PRO.to_string(),
            FLUX_1_DEV.to_string(),
            FLUX_1_SCHNELL.to_string(),
            FLUX_1_SCHNELL_PRO.to_string(),
            FLUX_1_DEV_LORA.to_string(),
            // Stable Diffusion models
            STABLE_DIFFUSION_3_5_LARGE.to_string(),
            STABLE_DIFFUSION_XL_BASE_1_0.to_string(),
            // Kolors
            KOLORS.to_string(),
        ]
    }

    /// Get all SiliconFlow models
    pub fn all_models() -> Vec<String> {
        let mut models = Vec::new();
        models.extend(all_chat_models());
        models.extend(all_embedding_models());
        models.extend(all_rerank_models());
        models.extend(all_image_models());
        models
    }
}

/// Groq model constants
pub mod groq {
    /// Llama 3.1 70B Versatile
    pub const LLAMA_3_1_70B: &str = "llama-3.1-70b-versatile";
    /// Llama 3.1 8B Instant
    pub const LLAMA_3_1_8B: &str = "llama-3.1-8b-instant";
    /// Mixtral 8x7B
    pub const MIXTRAL_8X7B: &str = "mixtral-8x7b-32768";

    /// Get all Groq models
    pub fn all_models() -> Vec<String> {
        vec![
            LLAMA_3_1_70B.to_string(),
            LLAMA_3_1_8B.to_string(),
            MIXTRAL_8X7B.to_string(),
        ]
    }
}

/// MoonshotAI model constants aligned with the audited AI SDK package subset.
pub mod moonshot {
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
}

/// Canonical AI SDK package-style MoonshotAI model namespace.
pub mod moonshotai {
    pub use super::moonshot::*;
}

/// Get models for a specific provider
pub fn get_models_for_provider(provider: &str) -> Vec<String> {
    match provider.to_lowercase().as_str() {
        "deepseek" => deepseek::all_models(),
        "deepinfra" => deepinfra::all_models(),
        "fireworks" => fireworks::all_models(),
        "openrouter" => openrouter::all_models(),
        "vertex-maas" => vertex_maas::all_models(),
        "xai" => xai::all_models(),
        "groq" => groq::all_models(),
        "siliconflow" => siliconflow::all_models(),
        "moonshot" | "moonshotai" => moonshot::all_models(),
        _ => vec![],
    }
}

/// Check if a model is supported by a provider
pub fn is_model_supported(provider: &str, model: &str) -> bool {
    get_models_for_provider(provider).contains(&model.to_string())
}

/// Model recommendations for different use cases
pub mod recommendations {
    use super::*;

    /// Recommended model for general chat
    pub const fn for_chat() -> &'static str {
        openrouter::openai::GPT_4O
    }

    /// Recommended model for coding tasks
    pub const fn for_coding() -> &'static str {
        deepseek::DEEPSEEK_V3_0324 // Use latest V3 model for coding
    }

    /// Recommended model for reasoning tasks
    pub const fn for_reasoning() -> &'static str {
        deepseek::REASONER
    }

    /// Recommended model for fast responses
    pub const fn for_fast_response() -> &'static str {
        groq::LLAMA_3_1_8B
    }

    /// Recommended model for cost-effective usage
    pub const fn for_cost_effective() -> &'static str {
        deepseek::CHAT
    }

    /// Recommended model for vision tasks
    pub const fn for_vision() -> &'static str {
        openrouter::openai::GPT_4O
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deepseek_models() {
        let models = deepseek::all_models();
        assert!(!models.is_empty());
        assert!(models.contains(&"deepseek-chat".to_string()));
    }

    #[test]
    fn test_openrouter_models() {
        let models = openrouter::all_models();
        assert!(!models.is_empty());
        assert!(models.contains(&"openai/gpt-4o".to_string()));
    }

    #[test]
    fn test_deepinfra_models() {
        let models = deepinfra::all_models();
        assert!(!models.is_empty());
        assert!(models.contains(&deepinfra::CHAT.to_string()));
        assert!(models.contains(&deepinfra::COMPLETION.to_string()));
        assert!(models.contains(&deepinfra::EMBEDDING.to_string()));
        assert!(models.contains(&deepinfra::IMAGE.to_string()));
    }

    #[test]
    fn test_vertex_maas_models() {
        let models = vertex_maas::all_models();
        assert!(!models.is_empty());
        assert!(models.contains(&vertex_maas::CHAT.to_string()));
        assert!(models.contains(&vertex_maas::COMPLETION.to_string()));
        assert!(models.contains(&vertex_maas::EMBEDDING.to_string()));
    }

    #[test]
    fn test_fireworks_models() {
        let models = fireworks::all_models();
        assert!(!models.is_empty());
        assert!(models.contains(&fireworks::CHAT.to_string()));
        assert!(models.contains(&fireworks::COMPLETION.to_string()));
        assert!(models.contains(&fireworks::EMBEDDING.to_string()));
        assert!(models.contains(&fireworks::IMAGE.to_string()));
    }

    #[test]
    fn test_get_models_for_provider() {
        let deepseek_models = get_models_for_provider("deepseek");
        assert!(!deepseek_models.is_empty());

        let deepinfra_models = get_models_for_provider("deepinfra");
        assert!(deepinfra_models.contains(&deepinfra::chat::LLAMA_V3P3_70B_INSTRUCT.to_string()));

        let vertex_maas_models = get_models_for_provider("vertex-maas");
        assert!(vertex_maas_models.contains(&vertex_maas::chat::DEEPSEEK_V3_2_MAAS.to_string()));

        let fireworks_models = get_models_for_provider("fireworks");
        assert!(fireworks_models.contains(&fireworks::chat::DEEPSEEK_V3.to_string()));

        let moonshotai_models = get_models_for_provider("moonshotai");
        assert!(moonshotai_models.contains(&moonshotai::KIMI_K2_0905.to_string()));

        let unknown_models = get_models_for_provider("unknown");
        assert!(unknown_models.is_empty());
    }

    #[test]
    fn test_is_model_supported() {
        assert!(is_model_supported("deepseek", "deepseek-chat"));
        assert!(is_model_supported(
            "deepinfra",
            deepinfra::image::FLUX_1_KONTEXT_PRO
        ));
        assert!(is_model_supported(
            "vertex-maas",
            vertex_maas::chat::QWEN3_NEXT_80B_A3B_THINKING_MAAS
        ));
        assert!(is_model_supported(
            "fireworks",
            fireworks::image::FLUX_KONTEXT_PRO
        ));
        assert!(is_model_supported("moonshotai", moonshotai::KIMI_K2P5));
        assert!(!is_model_supported("deepseek", "unknown-model"));
        assert!(!is_model_supported("unknown", "any-model"));
    }
}
