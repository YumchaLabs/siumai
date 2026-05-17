//! SiliconFlow model constants
//!
//! These constants are based on the actual models available from SiliconFlow API.
//! Updated to reflect the current model offerings as of the latest API query.
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
