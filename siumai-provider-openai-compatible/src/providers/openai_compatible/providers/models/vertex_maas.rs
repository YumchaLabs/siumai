//! Vertex MaaS model constants aligned with the audited AI SDK package subset.
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
