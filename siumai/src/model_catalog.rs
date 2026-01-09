//! Unified model catalogs and constants across providers.
//!
//! This module aggregates provider-owned model constant sets into a single,
//! ergonomic facade surface.
//!
//! Public entry points:
//! - `siumai::models`     → simplified constants (`model_constants` module)
//! - `siumai::constants`  → detailed provider catalogs (`constants` module)

/// Unified model constants for detailed access across providers.
pub mod constants {
    /// Re-export OpenAI model constants (detailed structure).
    #[cfg(feature = "openai")]
    pub use siumai_provider_openai::providers::openai::model_constants as openai;

    /// Re-export Anthropic model constants (detailed structure).
    #[cfg(feature = "anthropic")]
    pub use siumai_provider_anthropic::providers::anthropic::model_constants as anthropic;

    /// Re-export Gemini model constants (detailed structure).
    #[cfg(feature = "google")]
    pub use siumai_provider_gemini::providers::gemini::model_constants as gemini;

    /// Re-export OpenAI-compatible provider model constants.
    #[cfg(feature = "openai")]
    pub use siumai_provider_openai_compatible::providers::openai_compatible::providers::models as openai_compatible;

    /// Re-export Ollama model constants (detailed structure).
    #[cfg(feature = "ollama")]
    pub use siumai_provider_ollama::providers::ollama::model_constants as ollama;

    /// Re-export xAI model constants (detailed structure).
    #[cfg(feature = "xai")]
    pub use siumai_provider_xai::providers::xai::models as xai;

    /// Re-export Groq model constants (detailed structure).
    #[cfg(feature = "groq")]
    pub use siumai_provider_groq::providers::groq::models as groq;

    /// Re-export MiniMaxi model constants (detailed structure).
    #[cfg(feature = "minimaxi")]
    pub use siumai_provider_minimaxi::providers::minimaxi::model_constants as minimaxi;

    /// Re-export DeepSeek model constants (detailed structure).
    #[cfg(feature = "deepseek")]
    pub use siumai_provider_deepseek::providers::deepseek::models as deepseek;
}

/// Simplified model constants for easy access across providers.
pub mod model_constants {
    /// OpenAI models with simplified access.
    #[cfg(feature = "openai")]
    pub mod openai {
        use siumai_provider_openai::providers::openai::model_constants as c;

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

    /// Anthropic models with simplified access.
    #[cfg(feature = "anthropic")]
    pub mod anthropic {
        use siumai_provider_anthropic::providers::anthropic::model_constants as c;

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

    /// Gemini models with simplified access.
    #[cfg(feature = "google")]
    pub mod gemini {
        use siumai_provider_gemini::providers::gemini::model_constants as c;

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

    /// OpenAI-compatible provider models.
    ///
    /// Note: this module intentionally mirrors the OpenAI provider crate's model catalog
    /// so that `siumai::models::openai_compatible::*` keeps working without depending on
    /// the legacy umbrella crate.
    #[cfg(feature = "openai")]
    pub mod openai_compatible {
        pub use siumai_provider_openai_compatible::providers::openai_compatible::providers::models::*;
    }

    /// Ollama models with simplified access.
    #[cfg(feature = "ollama")]
    pub mod ollama {
        use siumai_provider_ollama::providers::ollama::model_constants as c;

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

    /// xAI models with simplified access.
    #[cfg(feature = "xai")]
    pub mod xai {
        pub use siumai_provider_xai::providers::xai::models::*;
    }

    /// Groq models with simplified access.
    #[cfg(feature = "groq")]
    pub mod groq {
        pub use siumai_provider_groq::providers::groq::models::*;
    }

    /// DeepSeek models with simplified access.
    #[cfg(feature = "deepseek")]
    pub mod deepseek {
        pub use siumai_provider_deepseek::providers::deepseek::models::*;
    }

    /// MiniMaxi models with simplified access.
    #[cfg(feature = "minimaxi")]
    pub mod minimaxi {
        use siumai_provider_minimaxi::providers::minimaxi::model_constants as c;

        // Text
        pub const MINIMAX_M2: &str = c::text::MINIMAX_M2;
        pub const MINIMAX_M2_STABLE: &str = c::text::MINIMAX_M2_STABLE;

        // Audio (TTS)
        pub const SPEECH_2_6_HD: &str = c::audio::SPEECH_2_6_HD;
        pub const SPEECH_2_6_TURBO: &str = c::audio::SPEECH_2_6_TURBO;

        // Voices (subset)
        pub const MALE_QN_QINGSE: &str = c::voice::MALE_QN_QINGSE;
        pub const FEMALE_SHAONV: &str = c::voice::FEMALE_SHAONV;

        // Images
        pub const IMAGE_01: &str = c::images::IMAGE_01;
        pub const IMAGE_01_LIVE: &str = c::images::IMAGE_01_LIVE;
    }
}
