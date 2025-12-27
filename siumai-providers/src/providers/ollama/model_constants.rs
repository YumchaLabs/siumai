//! Ollama Model Constants
//!
//! This module provides convenient constants for popular Ollama models, making it easy
//! for developers to reference specific models without hardcoding strings.
//!
//! # Popular Models
//!
//! Based on the most commonly used models in the Ollama ecosystem:
//! - **Llama 3.2**: Latest general-purpose models with various sizes
//! - **Llama 3.1**: Previous generation with proven performance
//! - **Code Llama**: Specialized for code generation and analysis
//! - **Mistral**: Alternative general-purpose models
//! - **Phi-3**: Microsoft's efficient small language models
//! - **Gemma**: Google's open models
//! - **Qwen2**: Alibaba's multilingual models
//! - **DeepSeek**: Specialized reasoning and coding models

/// Llama 3.2 model family constants (latest generation)
pub mod llama_3_2 {
    /// Llama 3.2 3B - Balanced performance and efficiency
    pub const LLAMA_3_2_3B: &str = "llama3.2:3b";
    /// Llama 3.2 1B - Ultra-fast, lightweight model
    pub const LLAMA_3_2_1B: &str = "llama3.2:1b";
    /// Llama 3.2 Latest - Default latest version
    pub const LLAMA_3_2_LATEST: &str = "llama3.2:latest";
    /// Llama 3.2 (alias for latest)
    pub const LLAMA_3_2: &str = "llama3.2";

    /// All Llama 3.2 models
    pub const ALL: &[&str] = &[LLAMA_3_2_3B, LLAMA_3_2_1B, LLAMA_3_2_LATEST, LLAMA_3_2];
}

/// Llama 3.1 model family constants (proven generation)
pub mod llama_3_1 {
    /// Llama 3.1 8B - Efficient general-purpose model
    pub const LLAMA_3_1_8B: &str = "llama3.1:8b";
    /// Llama 3.1 70B - High-capability model
    pub const LLAMA_3_1_70B: &str = "llama3.1:70b";
    /// Llama 3.1 405B - Largest model (requires significant resources)
    pub const LLAMA_3_1_405B: &str = "llama3.1:405b";
    /// Llama 3.1 Latest - Default latest version
    pub const LLAMA_3_1_LATEST: &str = "llama3.1:latest";
    /// Llama 3.1 (alias for latest)
    pub const LLAMA_3_1: &str = "llama3.1";

    /// All Llama 3.1 models
    pub const ALL: &[&str] = &[
        LLAMA_3_1_8B,
        LLAMA_3_1_70B,
        LLAMA_3_1_405B,
        LLAMA_3_1_LATEST,
        LLAMA_3_1,
    ];
}

/// Code Llama model family constants (specialized for coding)
pub mod code_llama {
    /// Code Llama 7B - Efficient code generation
    pub const CODE_LLAMA_7B: &str = "codellama:7b";
    /// Code Llama 13B - Balanced code generation
    pub const CODE_LLAMA_13B: &str = "codellama:13b";
    /// Code Llama 34B - High-quality code generation
    pub const CODE_LLAMA_34B: &str = "codellama:34b";
    /// Code Llama Latest - Default latest version
    pub const CODE_LLAMA_LATEST: &str = "codellama:latest";
    /// Code Llama (alias for latest)
    pub const CODE_LLAMA: &str = "codellama";

    /// All Code Llama models
    pub const ALL: &[&str] = &[
        CODE_LLAMA_7B,
        CODE_LLAMA_13B,
        CODE_LLAMA_34B,
        CODE_LLAMA_LATEST,
        CODE_LLAMA,
    ];
}

/// Mistral model family constants
pub mod mistral {
    /// Mistral 7B - Efficient general-purpose model
    pub const MISTRAL_7B: &str = "mistral:7b";
    /// Mistral Latest - Default latest version
    pub const MISTRAL_LATEST: &str = "mistral:latest";
    /// Mistral (alias for latest)
    pub const MISTRAL: &str = "mistral";

    /// All Mistral models
    pub const ALL: &[&str] = &[MISTRAL_7B, MISTRAL_LATEST, MISTRAL];
}

/// Phi-3 model family constants (Microsoft's efficient models)
pub mod phi_3 {
    /// Phi-3 Mini - Ultra-efficient small model
    pub const PHI_3_MINI: &str = "phi3:mini";
    /// Phi-3 Medium - Balanced efficiency and capability
    pub const PHI_3_MEDIUM: &str = "phi3:medium";
    /// Phi-3 Latest - Default latest version
    pub const PHI_3_LATEST: &str = "phi3:latest";
    /// Phi-3 (alias for latest)
    pub const PHI_3: &str = "phi3";

    /// All Phi-3 models
    pub const ALL: &[&str] = &[PHI_3_MINI, PHI_3_MEDIUM, PHI_3_LATEST, PHI_3];
}

/// Gemma model family constants (Google's open models)
pub mod gemma {
    /// Gemma 2B - Ultra-lightweight model
    pub const GEMMA_2B: &str = "gemma:2b";
    /// Gemma 7B - Standard model
    pub const GEMMA_7B: &str = "gemma:7b";
    /// Gemma Latest - Default latest version
    pub const GEMMA_LATEST: &str = "gemma:latest";
    /// Gemma (alias for latest)
    pub const GEMMA: &str = "gemma";

    /// All Gemma models
    pub const ALL: &[&str] = &[GEMMA_2B, GEMMA_7B, GEMMA_LATEST, GEMMA];
}

/// Qwen2 model family constants (Alibaba's multilingual models)
pub mod qwen2 {
    /// Qwen2 0.5B - Ultra-lightweight model
    pub const QWEN2_0_5B: &str = "qwen2:0.5b";
    /// Qwen2 1.5B - Lightweight model
    pub const QWEN2_1_5B: &str = "qwen2:1.5b";
    /// Qwen2 7B - Standard model
    pub const QWEN2_7B: &str = "qwen2:7b";
    /// Qwen2 72B - High-capability model
    pub const QWEN2_72B: &str = "qwen2:72b";
    /// Qwen2 Latest - Default latest version
    pub const QWEN2_LATEST: &str = "qwen2:latest";
    /// Qwen2 (alias for latest)
    pub const QWEN2: &str = "qwen2";

    /// All Qwen2 models
    pub const ALL: &[&str] = &[
        QWEN2_0_5B,
        QWEN2_1_5B,
        QWEN2_7B,
        QWEN2_72B,
        QWEN2_LATEST,
        QWEN2,
    ];
}

/// DeepSeek model family constants (specialized reasoning and coding)
pub mod deepseek {
    /// DeepSeek Coder - Specialized for code generation
    pub const DEEPSEEK_CODER: &str = "deepseek-coder:latest";
    /// DeepSeek R1 - Advanced reasoning model
    pub const DEEPSEEK_R1: &str = "deepseek-r1:latest";
    /// DeepSeek R1 (short alias)
    pub const DEEPSEEK_R1_SHORT: &str = "deepseek-r1";

    /// All DeepSeek models
    pub const ALL: &[&str] = &[DEEPSEEK_CODER, DEEPSEEK_R1, DEEPSEEK_R1_SHORT];
}

/// Embedding model constants
pub mod embeddings {
    /// Nomic Embed Text - High-quality text embeddings
    pub const NOMIC_EMBED_TEXT: &str = "nomic-embed-text";
    /// All MiniLM - Lightweight embeddings
    pub const ALL_MINILM: &str = "all-minilm";
    /// MXBAI Embed Large - Large embedding model
    pub const MXBAI_EMBED_LARGE: &str = "mxbai-embed-large";
    /// Snowflake Arctic Embed - Arctic embeddings
    pub const SNOWFLAKE_ARCTIC_EMBED: &str = "snowflake-arctic-embed";

    /// All embedding models
    pub const ALL: &[&str] = &[
        NOMIC_EMBED_TEXT,
        ALL_MINILM,
        MXBAI_EMBED_LARGE,
        SNOWFLAKE_ARCTIC_EMBED,
    ];
}

/// Popular model recommendations
pub mod popular {
    use super::*;

    /// Most balanced general-purpose model
    pub const GENERAL_PURPOSE: &str = llama_3_2::LLAMA_3_2_3B;
    /// Fastest lightweight model
    pub const LIGHTWEIGHT: &str = llama_3_2::LLAMA_3_2_1B;
    /// Best for coding tasks
    pub const CODING: &str = code_llama::CODE_LLAMA_13B;
    /// Best for reasoning tasks
    pub const REASONING: &str = deepseek::DEEPSEEK_R1;
    /// Most efficient small model
    pub const EFFICIENT: &str = phi_3::PHI_3_MINI;
    /// Best for embeddings
    pub const EMBEDDINGS: &str = embeddings::NOMIC_EMBED_TEXT;
    /// Latest flagship model
    pub const FLAGSHIP: &str = llama_3_2::LLAMA_3_2_LATEST;
}

pub use code_llama::CODE_LLAMA;
pub use deepseek::DEEPSEEK_R1;
pub use embeddings::NOMIC_EMBED_TEXT;
/// Simplified access to popular models (top-level constants)
pub use llama_3_2::LLAMA_3_2;
pub use llama_3_2::LLAMA_3_2_1B;
pub use llama_3_2::LLAMA_3_2_3B;
pub use mistral::MISTRAL;
