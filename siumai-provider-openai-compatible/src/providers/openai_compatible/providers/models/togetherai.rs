//! TogetherAI model constants aligned with the audited AI SDK package subset.
/// TogetherAI chat/language-model constants.
pub mod chat {
    pub const LLAMA_3_3_70B_INSTRUCT_TURBO: &str = "meta-llama/Llama-3.3-70B-Instruct-Turbo";
    pub const META_LLAMA_3_1_8B_INSTRUCT_TURBO: &str =
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo";
    pub const META_LLAMA_3_1_70B_INSTRUCT_TURBO: &str =
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo";
    pub const META_LLAMA_3_1_405B_INSTRUCT_TURBO: &str =
        "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo";
    pub const META_LLAMA_3_8B_INSTRUCT_TURBO: &str = "meta-llama/Meta-Llama-3-8B-Instruct-Turbo";
    pub const META_LLAMA_3_70B_INSTRUCT_TURBO: &str = "meta-llama/Meta-Llama-3-70B-Instruct-Turbo";
    pub const LLAMA_3_2_3B_INSTRUCT_TURBO: &str = "meta-llama/Llama-3.2-3B-Instruct-Turbo";
    pub const META_LLAMA_3_8B_INSTRUCT_LITE: &str = "meta-llama/Meta-Llama-3-8B-Instruct-Lite";
    pub const META_LLAMA_3_70B_INSTRUCT_LITE: &str = "meta-llama/Meta-Llama-3-70B-Instruct-Lite";
    pub const LLAMA_3_8B_CHAT_HF: &str = "meta-llama/Llama-3-8b-chat-hf";
    pub const LLAMA_3_70B_CHAT_HF: &str = "meta-llama/Llama-3-70b-chat-hf";
    pub const NEMOTRON_70B_INSTRUCT_HF: &str = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF";
    pub const QWEN_2_5_CODER_32B_INSTRUCT: &str = "Qwen/Qwen2.5-Coder-32B-Instruct";
    pub const QWQ_32B_PREVIEW: &str = "Qwen/QwQ-32B-Preview";
    pub const WIZARD_LM_2_8X22B: &str = "microsoft/WizardLM-2-8x22B";
    pub const GEMMA_2_27B_IT: &str = "google/gemma-2-27b-it";
    pub const GEMMA_2_9B_IT: &str = "google/gemma-2-9b-it";
    pub const DBRX_INSTRUCT: &str = "databricks/dbrx-instruct";
    pub const DEEPSEEK_LLM_67B_CHAT: &str = "deepseek-ai/deepseek-llm-67b-chat";
    pub const DEEPSEEK_V3: &str = "deepseek-ai/DeepSeek-V3";
    pub const GEMMA_2B_IT: &str = "google/gemma-2b-it";
    pub const MYTHOMAX_L2_13B: &str = "Gryphe/MythoMax-L2-13b";
    pub const LLAMA_2_13B_CHAT_HF: &str = "meta-llama/Llama-2-13b-chat-hf";
    pub const MISTRAL_7B_INSTRUCT_V0_1: &str = "mistralai/Mistral-7B-Instruct-v0.1";
    pub const MISTRAL_7B_INSTRUCT_V0_2: &str = "mistralai/Mistral-7B-Instruct-v0.2";
    pub const MISTRAL_7B_INSTRUCT_V0_3: &str = "mistralai/Mistral-7B-Instruct-v0.3";
    pub const MIXTRAL_8X7B_INSTRUCT_V0_1: &str = "mistralai/Mixtral-8x7B-Instruct-v0.1";
    pub const MIXTRAL_8X22B_INSTRUCT_V0_1: &str = "mistralai/Mixtral-8x22B-Instruct-v0.1";
    pub const NOUS_HERMES_2_MIXTRAL_8X7B_DPO: &str = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO";
    pub const QWEN_2_5_7B_INSTRUCT_TURBO: &str = "Qwen/Qwen2.5-7B-Instruct-Turbo";
    pub const QWEN_2_5_72B_INSTRUCT_TURBO: &str = "Qwen/Qwen2.5-72B-Instruct-Turbo";
    pub const QWEN_2_72B_INSTRUCT: &str = "Qwen/Qwen2-72B-Instruct";
    pub const SOLAR_10_7B_INSTRUCT_V1_0: &str = "upstage/SOLAR-10.7B-Instruct-v1.0";
}

/// TogetherAI completion-model constants.
pub mod completion {
    pub const LLAMA_2_70B_HF: &str = "meta-llama/Llama-2-70b-hf";
    pub const MISTRAL_7B_V0_1: &str = "mistralai/Mistral-7B-v0.1";
    pub const MIXTRAL_8X7B_V0_1: &str = "mistralai/Mixtral-8x7B-v0.1";
    pub const LLAMA_GUARD_7B: &str = "Meta-Llama/Llama-Guard-7b";
    pub const CODELLAMA_34B_INSTRUCT_HF: &str = "codellama/CodeLlama-34b-Instruct-hf";
    pub const QWEN_2_5_CODER_32B_INSTRUCT: &str = "Qwen/Qwen2.5-Coder-32B-Instruct";
}

/// TogetherAI embedding-model constants.
pub mod embedding {
    pub const M2_BERT_80M_2K_RETRIEVAL: &str = "togethercomputer/m2-bert-80M-2k-retrieval";
    pub const M2_BERT_80M_32K_RETRIEVAL: &str = "togethercomputer/m2-bert-80M-32k-retrieval";
    pub const M2_BERT_80M_8K_RETRIEVAL: &str = "togethercomputer/m2-bert-80M-8k-retrieval";
    pub const UAE_LARGE_V1: &str = "WhereIsAI/UAE-Large-V1";
    pub const BGE_LARGE_EN_V1_5: &str = "BAAI/bge-large-en-v1.5";
    pub const BGE_BASE_EN_V1_5: &str = "BAAI/bge-base-en-v1.5";
    pub const MSMARCO_BERT_BASE_DOT_V5: &str = "sentence-transformers/msmarco-bert-base-dot-v5";
    pub const BERT_BASE_UNCASED: &str = "bert-base-uncased";
}

/// TogetherAI image-model constants.
pub mod image {
    pub const STABLE_DIFFUSION_XL_BASE_1_0: &str = "stabilityai/stable-diffusion-xl-base-1.0";
    pub const FLUX_1_DEV: &str = "black-forest-labs/FLUX.1-dev";
    pub const FLUX_1_DEV_LORA: &str = "black-forest-labs/FLUX.1-dev-lora";
    pub const FLUX_1_SCHNELL: &str = "black-forest-labs/FLUX.1-schnell";
    pub const FLUX_1_CANNY: &str = "black-forest-labs/FLUX.1-canny";
    pub const FLUX_1_DEPTH: &str = "black-forest-labs/FLUX.1-depth";
    pub const FLUX_1_REDUX: &str = "black-forest-labs/FLUX.1-redux";
    pub const FLUX_1_1_PRO: &str = "black-forest-labs/FLUX.1.1-pro";
    pub const FLUX_1_PRO: &str = "black-forest-labs/FLUX.1-pro";
    pub const FLUX_1_SCHNELL_FREE: &str = "black-forest-labs/FLUX.1-schnell-Free";
    pub const FLUX_1_KONTEXT_PRO: &str = "black-forest-labs/FLUX.1-kontext-pro";
    pub const FLUX_1_KONTEXT_MAX: &str = "black-forest-labs/FLUX.1-kontext-max";
    pub const FLUX_1_KONTEXT_DEV: &str = "black-forest-labs/FLUX.1-kontext-dev";
}

/// TogetherAI rerank-model constants.
pub mod rerank {
    pub const LLAMA_RANK_V1: &str = "Salesforce/Llama-Rank-v1";
    pub const MXBAI_RERANK_LARGE_V2: &str = "mixedbread-ai/Mxbai-Rerank-Large-V2";
}

pub const CHAT: &str = chat::META_LLAMA_3_1_8B_INSTRUCT_TURBO;
pub const COMPLETION: &str = completion::QWEN_2_5_CODER_32B_INSTRUCT;
pub const EMBEDDING: &str = embedding::M2_BERT_80M_8K_RETRIEVAL;
pub const IMAGE: &str = image::FLUX_1_SCHNELL;
pub const RERANK: &str = rerank::LLAMA_RANK_V1;

pub const ALL_CHAT: &[&str] = &[
    chat::LLAMA_3_3_70B_INSTRUCT_TURBO,
    chat::META_LLAMA_3_1_8B_INSTRUCT_TURBO,
    chat::META_LLAMA_3_1_70B_INSTRUCT_TURBO,
    chat::META_LLAMA_3_1_405B_INSTRUCT_TURBO,
    chat::META_LLAMA_3_8B_INSTRUCT_TURBO,
    chat::META_LLAMA_3_70B_INSTRUCT_TURBO,
    chat::LLAMA_3_2_3B_INSTRUCT_TURBO,
    chat::META_LLAMA_3_8B_INSTRUCT_LITE,
    chat::META_LLAMA_3_70B_INSTRUCT_LITE,
    chat::LLAMA_3_8B_CHAT_HF,
    chat::LLAMA_3_70B_CHAT_HF,
    chat::NEMOTRON_70B_INSTRUCT_HF,
    chat::QWEN_2_5_CODER_32B_INSTRUCT,
    chat::QWQ_32B_PREVIEW,
    chat::WIZARD_LM_2_8X22B,
    chat::GEMMA_2_27B_IT,
    chat::GEMMA_2_9B_IT,
    chat::DBRX_INSTRUCT,
    chat::DEEPSEEK_LLM_67B_CHAT,
    chat::DEEPSEEK_V3,
    chat::GEMMA_2B_IT,
    chat::MYTHOMAX_L2_13B,
    chat::LLAMA_2_13B_CHAT_HF,
    chat::MISTRAL_7B_INSTRUCT_V0_1,
    chat::MISTRAL_7B_INSTRUCT_V0_2,
    chat::MISTRAL_7B_INSTRUCT_V0_3,
    chat::MIXTRAL_8X7B_INSTRUCT_V0_1,
    chat::MIXTRAL_8X22B_INSTRUCT_V0_1,
    chat::NOUS_HERMES_2_MIXTRAL_8X7B_DPO,
    chat::QWEN_2_5_7B_INSTRUCT_TURBO,
    chat::QWEN_2_5_72B_INSTRUCT_TURBO,
    chat::QWEN_2_72B_INSTRUCT,
    chat::SOLAR_10_7B_INSTRUCT_V1_0,
];

pub const ALL_COMPLETION: &[&str] = &[
    completion::LLAMA_2_70B_HF,
    completion::MISTRAL_7B_V0_1,
    completion::MIXTRAL_8X7B_V0_1,
    completion::LLAMA_GUARD_7B,
    completion::CODELLAMA_34B_INSTRUCT_HF,
    completion::QWEN_2_5_CODER_32B_INSTRUCT,
];

pub const ALL_EMBEDDING: &[&str] = &[
    embedding::M2_BERT_80M_2K_RETRIEVAL,
    embedding::M2_BERT_80M_32K_RETRIEVAL,
    embedding::M2_BERT_80M_8K_RETRIEVAL,
    embedding::UAE_LARGE_V1,
    embedding::BGE_LARGE_EN_V1_5,
    embedding::BGE_BASE_EN_V1_5,
    embedding::MSMARCO_BERT_BASE_DOT_V5,
    embedding::BERT_BASE_UNCASED,
];

pub const ALL_IMAGE: &[&str] = &[
    image::STABLE_DIFFUSION_XL_BASE_1_0,
    image::FLUX_1_DEV,
    image::FLUX_1_DEV_LORA,
    image::FLUX_1_SCHNELL,
    image::FLUX_1_CANNY,
    image::FLUX_1_DEPTH,
    image::FLUX_1_REDUX,
    image::FLUX_1_1_PRO,
    image::FLUX_1_PRO,
    image::FLUX_1_SCHNELL_FREE,
    image::FLUX_1_KONTEXT_PRO,
    image::FLUX_1_KONTEXT_MAX,
    image::FLUX_1_KONTEXT_DEV,
];

pub const ALL_RERANK: &[&str] = &[rerank::LLAMA_RANK_V1, rerank::MXBAI_RERANK_LARGE_V2];

pub fn all_models() -> Vec<String> {
    let mut models = Vec::new();
    models.extend(ALL_CHAT.iter().map(|&model| model.to_string()));
    models.extend(ALL_COMPLETION.iter().map(|&model| model.to_string()));
    models.extend(ALL_EMBEDDING.iter().map(|&model| model.to_string()));
    models.extend(ALL_IMAGE.iter().map(|&model| model.to_string()));
    models.extend(ALL_RERANK.iter().map(|&model| model.to_string()));
    models
}
