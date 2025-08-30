//! Model information and capabilities
//!
//! This module provides model information structures and unified access to model constants
//! from all providers for convenient developer usage.

/// Model information
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model ID
    pub id: String,
    /// Model name
    pub name: Option<String>,
    /// Model description
    pub description: Option<String>,
    /// Model owner/organization
    pub owned_by: String,
    /// Creation timestamp
    pub created: Option<u64>,
    /// Model capabilities
    pub capabilities: Vec<String>,
    /// Context window size
    pub context_window: Option<u32>,
    /// Maximum output tokens
    pub max_output_tokens: Option<u32>,
    /// Input cost per token
    pub input_cost_per_token: Option<f64>,
    /// Output cost per token
    pub output_cost_per_token: Option<f64>,
}

/// Unified model constants for easy access across all providers
///
/// This module provides convenient access to model constants from all supported providers,
/// making it easy for developers to reference specific models without hardcoding strings.
///
/// # Examples
///
/// ```rust
/// use siumai::{models, constants};
///
/// // Simple access to popular models
/// let openai_model = models::openai::GPT_4O;
/// let anthropic_model = models::anthropic::CLAUDE_OPUS_4_1;
/// let gemini_model = models::gemini::GEMINI_2_5_PRO;
///
/// // Access popular recommendations
/// let flagship_models = [
///     constants::popular::flagship::OPENAI,
///     constants::popular::flagship::ANTHROPIC,
///     constants::popular::flagship::GEMINI,
/// ];
/// ```
pub mod constants {
    /// Re-export OpenAI model constants (detailed structure)
    #[cfg(feature = "openai")]
    pub use crate::providers::openai::model_constants as openai;

    /// Re-export Anthropic model constants (detailed structure)
    #[cfg(feature = "anthropic")]
    pub use crate::providers::anthropic::model_constants as anthropic;

    /// Re-export Gemini model constants (detailed structure)
    #[cfg(feature = "google")]
    pub use crate::providers::gemini::model_constants as gemini;

    /// Re-export OpenAI-compatible provider model constants
    #[cfg(feature = "openai")]
    pub use crate::providers::openai_compatible::providers::models as openai_compatible;

    /// Re-export Ollama model constants (detailed structure)
    #[cfg(feature = "ollama")]
    pub use crate::providers::ollama::model_constants as ollama;

    /// Re-export xAI model constants (detailed structure)
    #[cfg(feature = "xai")]
    pub use crate::providers::xai::models as xai;

    /// Re-export Groq model constants (detailed structure)
    #[cfg(feature = "groq")]
    pub use crate::providers::groq::models as groq;

    /// Popular models across all providers
    ///
    /// This module provides curated selections of popular models from each provider,
    /// organized by use case (flagship, balanced, reasoning, etc.).
    pub mod popular {
        /// Most capable models from each provider
        pub mod flagship {
            /// OpenAI's most capable model
            #[cfg(feature = "openai")]
            pub const OPENAI: &str = super::super::openai::popular::FLAGSHIP;
            /// Anthropic's most capable model
            #[cfg(feature = "anthropic")]
            pub const ANTHROPIC: &str = super::super::anthropic::popular::FLAGSHIP;
            /// Google's most capable model
            #[cfg(feature = "google")]
            pub const GEMINI: &str = super::super::gemini::popular::FLAGSHIP;
            /// xAI's most capable model
            #[cfg(feature = "xai")]
            pub const XAI: &str = super::super::xai::popular::FLAGSHIP;
            /// Groq's most capable model
            #[cfg(feature = "groq")]
            pub const GROQ: &str = super::super::groq::popular::FLAGSHIP;
        }

        /// Best balanced models (capability vs cost)
        pub mod balanced {
            /// OpenAI's balanced model
            #[cfg(feature = "openai")]
            pub const OPENAI: &str = super::super::openai::popular::BALANCED;
            /// Anthropic's balanced model
            #[cfg(feature = "anthropic")]
            pub const ANTHROPIC: &str = super::super::anthropic::popular::BALANCED;
            /// Google's balanced model
            #[cfg(feature = "google")]
            pub const GEMINI: &str = super::super::gemini::popular::BALANCED;
            /// xAI's balanced model
            #[cfg(feature = "xai")]
            pub const XAI: &str = super::super::xai::popular::BALANCED;
            /// Groq's balanced model
            #[cfg(feature = "groq")]
            pub const GROQ: &str = super::super::groq::popular::BALANCED;
        }

        /// Best reasoning models
        pub mod reasoning {
            /// OpenAI's reasoning model
            #[cfg(feature = "openai")]
            pub const OPENAI: &str = super::super::openai::popular::REASONING;
            /// Anthropic's thinking model
            #[cfg(feature = "anthropic")]
            pub const ANTHROPIC: &str = super::super::anthropic::popular::THINKING;
            /// Google's flagship model (has thinking)
            #[cfg(feature = "google")]
            pub const GEMINI: &str = super::super::gemini::popular::FLAGSHIP;
            /// Ollama's reasoning model
            #[cfg(feature = "ollama")]
            pub const OLLAMA: &str = super::super::ollama::popular::REASONING;
            /// xAI's reasoning model
            #[cfg(feature = "xai")]
            pub const XAI: &str = super::super::xai::popular::REASONING;
            /// Groq's reasoning model
            #[cfg(feature = "groq")]
            pub const GROQ: &str = super::super::groq::popular::REASONING;
        }

        /// Most economical models
        pub mod economical {
            /// OpenAI's economical model
            #[cfg(feature = "openai")]
            pub const OPENAI: &str = super::super::openai::popular::ECONOMICAL;
            /// Anthropic's fast model
            #[cfg(feature = "anthropic")]
            pub const ANTHROPIC: &str = super::super::anthropic::popular::FAST;
            /// Google's economical model
            #[cfg(feature = "google")]
            pub const GEMINI: &str = super::super::gemini::popular::ECONOMICAL;
            /// Ollama's lightweight model (free local)
            #[cfg(feature = "ollama")]
            pub const OLLAMA: &str = super::super::ollama::popular::LIGHTWEIGHT;
            /// xAI's lightweight model
            #[cfg(feature = "xai")]
            pub const XAI: &str = super::super::xai::popular::LIGHTWEIGHT;
            /// Groq's lightweight model
            #[cfg(feature = "groq")]
            pub const GROQ: &str = super::super::groq::popular::LIGHTWEIGHT;
        }

        /// Latest and most advanced models
        pub mod latest {
            /// OpenAI's latest model
            #[cfg(feature = "openai")]
            pub const OPENAI: &str = super::super::openai::popular::LATEST;
            /// Anthropic's latest model
            #[cfg(feature = "anthropic")]
            pub const ANTHROPIC: &str = super::super::anthropic::popular::LATEST;
            /// Google's latest model
            #[cfg(feature = "google")]
            pub const GEMINI: &str = super::super::gemini::popular::LATEST;
            /// xAI's latest model
            #[cfg(feature = "xai")]
            pub const XAI: &str = super::super::xai::popular::LATEST;
            /// Groq's latest model
            #[cfg(feature = "groq")]
            pub const GROQ: &str = super::super::groq::popular::LATEST;
        }
    }

    /// Get all available chat models from all providers
    pub fn all_chat_models() -> Vec<&'static str> {
        let mut models = Vec::new();
        #[cfg(feature = "openai")]
        models.extend_from_slice(&openai::all_chat_models());
        #[cfg(feature = "anthropic")]
        models.extend_from_slice(&anthropic::all_chat_models());
        #[cfg(feature = "google")]
        models.extend_from_slice(&gemini::all_chat_models());
        models
    }

    /// Get all reasoning models from all providers
    pub fn all_reasoning_models() -> Vec<&'static str> {
        let mut models = Vec::new();
        #[cfg(feature = "openai")]
        models.extend_from_slice(&openai::all_reasoning_models());
        #[cfg(feature = "anthropic")]
        models.extend_from_slice(&anthropic::all_thinking_models());
        #[cfg(feature = "google")]
        models.extend_from_slice(&gemini::all_thinking_models());
        models
    }

    /// Get all multimodal models from all providers
    pub fn all_multimodal_models() -> Vec<&'static str> {
        let mut models = Vec::new();
        #[cfg(feature = "openai")]
        models.extend_from_slice(&openai::all_multimodal_models());
        #[cfg(feature = "anthropic")]
        models.extend_from_slice(&anthropic::all_vision_models());
        #[cfg(feature = "google")]
        models.extend_from_slice(&gemini::all_chat_models()); // Most Gemini models are multimodal
        models
    }

    /// Get all audio generation models from all providers
    pub fn all_audio_generation_models() -> Vec<&'static str> {
        let mut models = Vec::new();
        #[cfg(feature = "openai")]
        models.extend_from_slice(openai::audio::ALL);
        #[cfg(feature = "google")]
        models.extend_from_slice(&gemini::all_audio_generation_models());
        models
    }

    /// Get all image generation models from all providers
    pub fn all_image_generation_models() -> Vec<&'static str> {
        let mut models = Vec::new();
        #[cfg(feature = "openai")]
        models.extend_from_slice(openai::images::ALL);
        #[cfg(feature = "google")]
        models.extend_from_slice(&gemini::all_image_generation_models());
        models
    }

    /// Get all embedding models from all providers
    pub fn all_embedding_models() -> Vec<&'static str> {
        #[cfg(feature = "openai")]
        return openai::embeddings::ALL.to_vec();

        #[cfg(not(feature = "openai"))]
        Vec::new()
    }
}

/// Simplified model constants with shorter namespaces
///
/// This module provides a more concise way to access model constants
/// with shorter namespaces for better developer experience.
///
/// # Examples
///
/// ```rust
/// use siumai::prelude::model_constants;
///
/// // Short and sweet access
/// let model = model_constants::openai::GPT_4O;
/// let claude = model_constants::anthropic::CLAUDE_OPUS_4_1;
/// let gemini = model_constants::gemini::GEMINI_2_5_PRO;
/// ```
pub mod model_constants {
    /// OpenAI models with simplified access
    #[cfg(feature = "openai")]
    pub mod openai {
        use crate::providers::openai::model_constants as c;

        // GPT-4o family
        pub const GPT_4O: &str = c::gpt_4o::GPT_4O;
        pub const GPT_4O_MINI: &str = c::gpt_4o::GPT_4O_MINI;
        pub const GPT_4O_AUDIO: &str = c::gpt_4o::GPT_4O_AUDIO_PREVIEW;

        // GPT-4.1 family
        pub const GPT_4_1: &str = c::gpt_4_1::GPT_4_1;
        pub const GPT_4_1_MINI: &str = c::gpt_4_1::GPT_4_1_MINI;
        pub const GPT_4_1_NANO: &str = c::gpt_4_1::GPT_4_1_NANO;

        // GPT-4.5 family
        pub const GPT_4_5: &str = c::gpt_4_5::GPT_4_5;
        pub const GPT_4_5_PREVIEW: &str = c::gpt_4_5::GPT_4_5_PREVIEW;

        // GPT-5 family
        pub const GPT_5: &str = c::gpt_5::GPT_5;
        pub const GPT_5_MINI: &str = c::gpt_5::GPT_5_MINI;
        pub const GPT_5_NANO: &str = c::gpt_5::GPT_5_NANO;

        // GPT-4 Turbo
        pub const GPT_4_TURBO: &str = c::gpt_4_turbo::GPT_4_TURBO;

        // GPT-4 Classic
        pub const GPT_4: &str = c::gpt_4::GPT_4;
        pub const GPT_4_32K: &str = c::gpt_4::GPT_4_32K;

        // o1 reasoning models
        pub const O1: &str = c::o1::O1;
        pub const O1_PREVIEW: &str = c::o1::O1_PREVIEW;
        pub const O1_MINI: &str = c::o1::O1_MINI;

        // o3 reasoning models
        pub const O3: &str = c::o3::O3;
        pub const O3_MINI: &str = c::o3::O3_MINI;

        // o4 reasoning models
        pub const O4_MINI: &str = c::o4::O4_MINI;

        // GPT-3.5
        pub const GPT_3_5_TURBO: &str = c::gpt_3_5::GPT_3_5_TURBO;

        // Audio models
        pub const TTS_1: &str = c::audio::TTS_1;
        pub const TTS_1_HD: &str = c::audio::TTS_1_HD;
        pub const WHISPER_1: &str = c::audio::WHISPER_1;

        // Image models
        pub const DALL_E_2: &str = c::images::DALL_E_2;
        pub const DALL_E_3: &str = c::images::DALL_E_3;

        // Embedding models
        pub const TEXT_EMBEDDING_3_SMALL: &str = c::embeddings::TEXT_EMBEDDING_3_SMALL;
        pub const TEXT_EMBEDDING_3_LARGE: &str = c::embeddings::TEXT_EMBEDDING_3_LARGE;
    }

    /// Anthropic models with simplified access
    #[cfg(feature = "anthropic")]
    pub mod anthropic {
        use crate::providers::anthropic::model_constants as c;

        // Claude Opus 4.1 (latest flagship)
        pub const CLAUDE_OPUS_4_1: &str = c::claude_opus_4_1::CLAUDE_OPUS_4_1;

        // Claude Opus 4
        pub const CLAUDE_OPUS_4: &str = c::claude_opus_4::CLAUDE_OPUS_4_20250514;

        // Claude Sonnet 4
        pub const CLAUDE_SONNET_4: &str = c::claude_sonnet_4::CLAUDE_SONNET_4_20250514;

        // Claude Sonnet 3.7 (thinking)
        pub const CLAUDE_SONNET_3_7: &str = c::claude_sonnet_3_7::CLAUDE_3_7_SONNET_20250219;

        // Claude Sonnet 3.5
        pub const CLAUDE_SONNET_3_5: &str = c::claude_sonnet_3_5::CLAUDE_3_5_SONNET_20241022;
        pub const CLAUDE_SONNET_3_5_LEGACY: &str = c::claude_sonnet_3_5::CLAUDE_3_5_SONNET_20240620;

        // Claude Haiku 3.5
        pub const CLAUDE_HAIKU_3_5: &str = c::claude_haiku_3_5::CLAUDE_3_5_HAIKU_20241022;

        // Claude 3 (legacy)
        pub const CLAUDE_OPUS_3: &str = c::claude_opus_3::CLAUDE_3_OPUS_20240229;
        pub const CLAUDE_SONNET_3: &str = c::claude_sonnet_3::CLAUDE_3_SONNET_20240229;
        pub const CLAUDE_HAIKU_3: &str = c::claude_haiku_3::CLAUDE_3_HAIKU_20240307;
    }

    /// Gemini models with simplified access
    #[cfg(feature = "google")]
    pub mod gemini {
        use crate::providers::gemini::model_constants as c;

        // Gemini 2.5 Pro (flagship)
        pub const GEMINI_2_5_PRO: &str = c::gemini_2_5_pro::GEMINI_2_5_PRO;

        // Gemini 2.5 Flash
        pub const GEMINI_2_5_FLASH: &str = c::gemini_2_5_flash::GEMINI_2_5_FLASH;

        // Gemini 2.5 Flash-Lite
        pub const GEMINI_2_5_FLASH_LITE: &str = c::gemini_2_5_flash_lite::GEMINI_2_5_FLASH_LITE;

        // Gemini 2.0 Flash
        pub const GEMINI_2_0_FLASH: &str = c::gemini_2_0_flash::GEMINI_2_0_FLASH;
        pub const GEMINI_2_0_FLASH_EXP: &str = c::gemini_2_0_flash::GEMINI_2_0_FLASH_EXP;

        // Gemini 2.0 Flash-Lite
        pub const GEMINI_2_0_FLASH_LITE: &str = c::gemini_2_0_flash_lite::GEMINI_2_0_FLASH_LITE;

        // Gemini 1.5 (legacy)
        pub const GEMINI_1_5_PRO: &str = c::gemini_1_5_pro::GEMINI_1_5_PRO;
        pub const GEMINI_1_5_FLASH: &str = c::gemini_1_5_flash::GEMINI_1_5_FLASH;
        pub const GEMINI_1_5_FLASH_8B: &str = c::gemini_1_5_flash_8b::GEMINI_1_5_FLASH_8B;

        // Live API models
        pub const GEMINI_LIVE_2_5_FLASH: &str =
            c::gemini_2_5_flash_live::GEMINI_LIVE_2_5_FLASH_PREVIEW;
        pub const GEMINI_LIVE_2_0_FLASH: &str = c::gemini_2_0_flash_live::GEMINI_2_0_FLASH_LIVE_001;
    }

    /// OpenAI-compatible provider models
    #[cfg(feature = "openai")]
    pub mod openai_compatible {
        use crate::providers::openai_compatible::providers::models as c;

        /// DeepSeek models
        pub mod deepseek {
            use super::c;

            pub const CHAT: &str = c::deepseek::CHAT;
            pub const REASONER: &str = c::deepseek::REASONER;
            pub const V3: &str = c::deepseek::DEEPSEEK_V3_0324;
            pub const R1: &str = c::deepseek::DEEPSEEK_R1_0528;
        }

        /// OpenRouter models (popular selections)
        pub mod openrouter {
            use super::c;

            pub const GPT_4O: &str = c::openrouter::openai::GPT_4O;
            pub const CLAUDE_3_5_SONNET: &str = c::openrouter::anthropic::CLAUDE_3_5_SONNET;
            pub const CLAUDE_OPUS_4_1: &str = c::openrouter::anthropic::CLAUDE_OPUS_4_1;
            pub const GEMINI_2_5_PRO: &str = c::openrouter::google::GEMINI_2_5_PRO;
            pub const DEEPSEEK_REASONER: &str = c::openrouter::deepseek::DEEPSEEK_REASONER;
            pub const LLAMA_3_1_405B: &str = c::openrouter::meta::LLAMA_3_1_405B;
        }

        /// SiliconFlow models
        pub mod siliconflow {
            use super::c;

            // ========================================================================
            // Chat Models - Most Popular and Recommended
            // ========================================================================

            /// DeepSeek V3.1 - Latest flagship reasoning model
            pub const DEEPSEEK_V3_1: &str = c::siliconflow::DEEPSEEK_V3_1;
            /// DeepSeek V3.1 Pro version
            pub const DEEPSEEK_V3_1_PRO: &str = c::siliconflow::DEEPSEEK_V3_1_PRO;
            /// DeepSeek V3 - Previous flagship model
            pub const DEEPSEEK_V3: &str = c::siliconflow::DEEPSEEK_V3;
            /// DeepSeek V3 Pro version
            pub const DEEPSEEK_V3_PRO: &str = c::siliconflow::DEEPSEEK_V3_PRO;
            /// DeepSeek R1 - Reasoning model
            pub const DEEPSEEK_R1: &str = c::siliconflow::DEEPSEEK_R1;
            /// DeepSeek R1 Pro version
            pub const DEEPSEEK_R1_PRO: &str = c::siliconflow::DEEPSEEK_R1_PRO;
            /// DeepSeek V2.5 - Previous generation
            pub const DEEPSEEK_V2_5: &str = c::siliconflow::DEEPSEEK_V2_5;
            /// DeepSeek VL2 - Vision-language model
            pub const DEEPSEEK_VL2: &str = c::siliconflow::DEEPSEEK_VL2;

            /// Qwen 3 235B A22B - Latest flagship
            pub const QWEN3_235B_A22B: &str = c::siliconflow::QWEN3_235B_A22B;
            /// Qwen 3 235B A22B Instruct
            pub const QWEN3_235B_A22B_INSTRUCT: &str = c::siliconflow::QWEN3_235B_A22B_INSTRUCT;
            /// Qwen 3 235B A22B Thinking
            pub const QWEN3_235B_A22B_THINKING: &str = c::siliconflow::QWEN3_235B_A22B_THINKING;
            /// Qwen 3 32B
            pub const QWEN3_32B: &str = c::siliconflow::QWEN3_32B;
            /// Qwen 3 30B A3B
            pub const QWEN3_30B_A3B: &str = c::siliconflow::QWEN3_30B_A3B;
            /// Qwen 3 30B A3B Instruct
            pub const QWEN3_30B_A3B_INSTRUCT: &str = c::siliconflow::QWEN3_30B_A3B_INSTRUCT;
            /// Qwen 3 30B A3B Thinking
            pub const QWEN3_30B_A3B_THINKING: &str = c::siliconflow::QWEN3_30B_A3B_THINKING;
            /// Qwen 3 14B
            pub const QWEN3_14B: &str = c::siliconflow::QWEN3_14B;
            /// Qwen 3 8B
            pub const QWEN3_8B: &str = c::siliconflow::QWEN3_8B;

            /// Qwen 2.5 72B Instruct
            pub const QWEN_2_5_72B_INSTRUCT: &str = c::siliconflow::QWEN_2_5_72B_INSTRUCT;
            /// Qwen 2.5 72B Instruct 128K context
            pub const QWEN_2_5_72B_INSTRUCT_128K: &str = c::siliconflow::QWEN_2_5_72B_INSTRUCT_128K;
            /// Qwen 2.5 32B Instruct
            pub const QWEN_2_5_32B_INSTRUCT: &str = c::siliconflow::QWEN_2_5_32B_INSTRUCT;
            /// Qwen 2.5 14B Instruct
            pub const QWEN_2_5_14B_INSTRUCT: &str = c::siliconflow::QWEN_2_5_14B_INSTRUCT;
            /// Qwen 2.5 7B Instruct
            pub const QWEN_2_5_7B_INSTRUCT: &str = c::siliconflow::QWEN_2_5_7B_INSTRUCT;

            /// Qwen 2.5 VL 72B Instruct - Vision-language model
            pub const QWEN_2_5_VL_72B_INSTRUCT: &str = c::siliconflow::QWEN_2_5_VL_72B_INSTRUCT;
            /// Qwen 2.5 VL 32B Instruct - Vision-language model
            pub const QWEN_2_5_VL_32B_INSTRUCT: &str = c::siliconflow::QWEN_2_5_VL_32B_INSTRUCT;
            /// Qwen 2.5 VL 7B Instruct Pro - Vision-language model
            pub const QWEN_2_5_VL_7B_INSTRUCT_PRO: &str =
                c::siliconflow::QWEN_2_5_VL_7B_INSTRUCT_PRO;

            /// Qwen 2.5 Coder models
            pub const QWEN_2_5_CODER_32B_INSTRUCT: &str =
                c::siliconflow::QWEN_2_5_CODER_32B_INSTRUCT;
            pub const QWEN_2_5_CODER_7B_INSTRUCT: &str = c::siliconflow::QWEN_2_5_CODER_7B_INSTRUCT;
            pub const QWEN_2_5_CODER_7B_INSTRUCT_PRO: &str =
                c::siliconflow::QWEN_2_5_CODER_7B_INSTRUCT_PRO;

            /// Qwen 3 Coder models
            pub const QWEN3_CODER_480B_A35B_INSTRUCT: &str =
                c::siliconflow::QWEN3_CODER_480B_A35B_INSTRUCT;
            pub const QWEN3_CODER_30B_A3B_INSTRUCT: &str =
                c::siliconflow::QWEN3_CODER_30B_A3B_INSTRUCT;

            /// QwQ 32B - Reasoning model
            pub const QWQ_32B: &str = c::siliconflow::QWQ_32B;
            /// QVQ 72B Preview - Vision-question model
            pub const QVQ_72B_PREVIEW: &str = c::siliconflow::QVQ_72B_PREVIEW;

            /// Kimi K2 Instruct
            pub const KIMI_K2_INSTRUCT: &str = c::siliconflow::KIMI_K2_INSTRUCT;
            /// Kimi K2 Instruct Pro
            pub const KIMI_K2_INSTRUCT_PRO: &str = c::siliconflow::KIMI_K2_INSTRUCT_PRO;

            /// GLM models
            pub const GLM_4_5: &str = c::siliconflow::GLM_4_5;
            pub const GLM_4_5_AIR: &str = c::siliconflow::GLM_4_5_AIR;
            pub const GLM_4_5V: &str = c::siliconflow::GLM_4_5V;
            pub const GLM_4_1V_9B_THINKING: &str = c::siliconflow::GLM_4_1V_9B_THINKING;
            pub const GLM_4_1V_9B_THINKING_PRO: &str = c::siliconflow::GLM_4_1V_9B_THINKING_PRO;

            // ========================================================================
            // Embedding Models
            // ========================================================================

            /// BCE Embedding Base V1 - NetEase Youdao (Recommended)
            pub const BCE_EMBEDDING_BASE_V1: &str = c::siliconflow::BCE_EMBEDDING_BASE_V1;

            /// Qwen 3 Embedding models (Latest)
            pub const QWEN3_EMBEDDING_8B: &str = c::siliconflow::QWEN3_EMBEDDING_8B;
            pub const QWEN3_EMBEDDING_4B: &str = c::siliconflow::QWEN3_EMBEDDING_4B;
            pub const QWEN3_EMBEDDING_0_6B: &str = c::siliconflow::QWEN3_EMBEDDING_0_6B;

            /// BGE models (Legacy but still available)
            pub const BGE_LARGE_EN_V1_5: &str = c::siliconflow::BGE_LARGE_EN_V1_5;
            pub const BGE_LARGE_ZH_V1_5: &str = c::siliconflow::BGE_LARGE_ZH_V1_5;
            pub const BGE_M3: &str = c::siliconflow::BGE_M3;
            pub const BGE_M3_PRO: &str = c::siliconflow::BGE_M3_PRO;

            // ========================================================================
            // Rerank Models
            // ========================================================================

            /// BGE Reranker V2 M3 (Recommended)
            pub const BGE_RERANKER_V2_M3: &str = c::siliconflow::BGE_RERANKER_V2_M3;
            /// BGE Reranker V2 M3 Pro
            pub const BGE_RERANKER_V2_M3_PRO: &str = c::siliconflow::BGE_RERANKER_V2_M3_PRO;

            /// BCE Reranker Base V1 - NetEase Youdao
            pub const BCE_RERANKER_BASE_V1: &str = c::siliconflow::BCE_RERANKER_BASE_V1;

            /// Qwen 3 Reranker models (Latest)
            pub const QWEN3_RERANKER_8B: &str = c::siliconflow::QWEN3_RERANKER_8B;
            pub const QWEN3_RERANKER_4B: &str = c::siliconflow::QWEN3_RERANKER_4B;
            pub const QWEN3_RERANKER_0_6B: &str = c::siliconflow::QWEN3_RERANKER_0_6B;

            // ========================================================================
            // Image Generation Models
            // ========================================================================

            /// FLUX.1 models (Recommended)
            pub const FLUX_1_PRO: &str = c::siliconflow::FLUX_1_PRO;
            pub const FLUX_1_DEV: &str = c::siliconflow::FLUX_1_DEV;
            pub const FLUX_1_SCHNELL: &str = c::siliconflow::FLUX_1_SCHNELL;
            pub const FLUX_1_SCHNELL_PRO: &str = c::siliconflow::FLUX_1_SCHNELL_PRO;
            pub const FLUX_1_DEV_LORA: &str = c::siliconflow::FLUX_1_DEV_LORA;

            /// Stable Diffusion models
            pub const STABLE_DIFFUSION_3_5_LARGE: &str = c::siliconflow::STABLE_DIFFUSION_3_5_LARGE;
            pub const STABLE_DIFFUSION_XL_BASE_1_0: &str =
                c::siliconflow::STABLE_DIFFUSION_XL_BASE_1_0;

            /// Kolors
            pub const KOLORS: &str = c::siliconflow::KOLORS;
        }
    }

    /// Ollama models with simplified access
    #[cfg(feature = "ollama")]
    pub mod ollama {
        use crate::providers::ollama::model_constants as c;

        // Llama 3.2 family
        pub const LLAMA_3_2: &str = c::llama_3_2::LLAMA_3_2;
        pub const LLAMA_3_2_3B: &str = c::llama_3_2::LLAMA_3_2_3B;
        pub const LLAMA_3_2_1B: &str = c::llama_3_2::LLAMA_3_2_1B;

        // Llama 3.1 family
        pub const LLAMA_3_1: &str = c::llama_3_1::LLAMA_3_1;
        pub const LLAMA_3_1_8B: &str = c::llama_3_1::LLAMA_3_1_8B;
        pub const LLAMA_3_1_70B: &str = c::llama_3_1::LLAMA_3_1_70B;

        // Code Llama
        pub const CODE_LLAMA: &str = c::code_llama::CODE_LLAMA;
        pub const CODE_LLAMA_13B: &str = c::code_llama::CODE_LLAMA_13B;

        // Other popular models
        pub const MISTRAL: &str = c::mistral::MISTRAL;
        pub const PHI_3: &str = c::phi_3::PHI_3;
        pub const GEMMA: &str = c::gemma::GEMMA;
        pub const QWEN2: &str = c::qwen2::QWEN2;

        // DeepSeek models
        pub const DEEPSEEK_R1: &str = c::deepseek::DEEPSEEK_R1;
        pub const DEEPSEEK_CODER: &str = c::deepseek::DEEPSEEK_CODER;

        // Embedding models
        pub const NOMIC_EMBED_TEXT: &str = c::embeddings::NOMIC_EMBED_TEXT;
    }

    /// xAI models with simplified access
    #[cfg(feature = "xai")]
    pub mod xai {
        use crate::providers::xai::models as c;

        // Grok 4 family (latest flagship)
        pub const GROK_4: &str = c::grok_4::GROK_4;
        pub const GROK_4_0709: &str = c::grok_4::GROK_4_0709;
        pub const GROK_4_LATEST: &str = c::grok_4::GROK_4_LATEST;

        // Grok 3 family
        pub const GROK_3: &str = c::grok_3::GROK_3;
        pub const GROK_3_LATEST: &str = c::grok_3::GROK_3_LATEST;
        pub const GROK_3_MINI: &str = c::grok_3::GROK_3_MINI;
        pub const GROK_3_FAST: &str = c::grok_3::GROK_3_FAST;

        // Grok 2 family
        pub const GROK_2: &str = c::grok_2::GROK_2;
        pub const GROK_2_LATEST: &str = c::grok_2::GROK_2_LATEST;

        // Image generation
        pub const GROK_2_IMAGE: &str = c::images::GROK_2_IMAGE;

        // Legacy
        pub const GROK_BETA: &str = c::legacy::GROK_BETA;
    }

    /// Groq models with simplified access
    #[cfg(feature = "groq")]
    pub mod groq {
        use crate::providers::groq::models as c;

        // Production models (stable)
        pub const LLAMA_3_1_8B_INSTANT: &str = c::production::LLAMA_3_1_8B_INSTANT;
        pub const LLAMA_3_3_70B_VERSATILE: &str = c::production::LLAMA_3_3_70B_VERSATILE;
        pub const LLAMA_GUARD_4_12B: &str = c::production::LLAMA_GUARD_4_12B;
        pub const WHISPER_LARGE_V3: &str = c::production::WHISPER_LARGE_V3;
        pub const WHISPER_LARGE_V3_TURBO: &str = c::production::WHISPER_LARGE_V3_TURBO;

        // Preview models (experimental)
        pub const DEEPSEEK_R1_DISTILL_LLAMA_70B: &str = c::preview::DEEPSEEK_R1_DISTILL_LLAMA_70B;
        pub const GPT_OSS_120B: &str = c::preview::GPT_OSS_120B;
        pub const GPT_OSS_20B: &str = c::preview::GPT_OSS_20B;
        pub const QWEN3_32B: &str = c::preview::QWEN3_32B;
        pub const PLAYAI_TTS: &str = c::preview::PLAYAI_TTS;

        // System models
        pub const COMPOUND_BETA: &str = c::systems::COMPOUND_BETA;
        pub const COMPOUND_BETA_MINI: &str = c::systems::COMPOUND_BETA_MINI;
    }
}
