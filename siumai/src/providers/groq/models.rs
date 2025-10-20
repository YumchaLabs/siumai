//! Groq Model Constants
//!
//! This module provides convenient constants for Groq models, making it easy
//! for developers to reference specific models without hardcoding strings.
//!
//! # Model Categories
//!
//! - **Production Models**: Stable models for production use
//! - **Preview Models**: Experimental models for evaluation
//! - **Audio Models**: Speech-to-text and text-to-speech models
//! - **Systems**: Compound systems with built-in tools

/// Production models - stable and reliable for production use
pub mod production {
    /// Llama 3.1 8B Instant - Fast and efficient model
    pub const LLAMA_3_1_8B_INSTANT: &str = "llama-3.1-8b-instant";
    /// Llama 3.3 70B Versatile - Balanced capability and performance
    pub const LLAMA_3_3_70B_VERSATILE: &str = "llama-3.3-70b-versatile";
    /// Llama Guard 4 12B - Content moderation model
    pub const LLAMA_GUARD_4_12B: &str = "meta-llama/llama-guard-4-12b";
    /// Whisper Large v3 - Speech-to-text model
    pub const WHISPER_LARGE_V3: &str = "whisper-large-v3";
    /// Whisper Large v3 Turbo - Faster speech-to-text model
    pub const WHISPER_LARGE_V3_TURBO: &str = "whisper-large-v3-turbo";

    /// All production models
    pub const ALL: &[&str] = &[
        LLAMA_3_1_8B_INSTANT,
        LLAMA_3_3_70B_VERSATILE,
        LLAMA_GUARD_4_12B,
        WHISPER_LARGE_V3,
        WHISPER_LARGE_V3_TURBO,
    ];
}

/// Preview models - experimental models for evaluation only
pub mod preview {
    /// DeepSeek R1 Distill Llama 70B - Reasoning model
    pub const DEEPSEEK_R1_DISTILL_LLAMA_70B: &str = "deepseek-r1-distill-llama-70b";
    /// Llama 4 Maverick 17B 128E Instruct - Experimental Llama 4 model
    pub const LLAMA_4_MAVERICK_17B_128E_INSTRUCT: &str =
        "meta-llama/llama-4-maverick-17b-128e-instruct";
    /// Llama 4 Scout 17B 16E Instruct - Experimental Llama 4 model
    pub const LLAMA_4_SCOUT_17B_16E_INSTRUCT: &str = "meta-llama/llama-4-scout-17b-16e-instruct";
    /// Llama Prompt Guard 2 22M - Small prompt guard model
    pub const LLAMA_PROMPT_GUARD_2_22M: &str = "meta-llama/llama-prompt-guard-2-22m";
    /// Llama Prompt Guard 2 86M - Larger prompt guard model
    pub const LLAMA_PROMPT_GUARD_2_86M: &str = "meta-llama/llama-prompt-guard-2-86m";
    /// Moonshot AI Kimi K2 Instruct - Moonshot AI model
    pub const KIMI_K2_INSTRUCT: &str = "moonshotai/kimi-k2-instruct";
    /// OpenAI GPT-OSS 120B - OpenAI's large open-source model
    pub const GPT_OSS_120B: &str = "openai/gpt-oss-120b";
    /// OpenAI GPT-OSS 20B - OpenAI's compact open-source model
    pub const GPT_OSS_20B: &str = "openai/gpt-oss-20b";
    /// PlayAI TTS - Text-to-speech model
    pub const PLAYAI_TTS: &str = "playai-tts";
    /// PlayAI TTS Arabic - Arabic text-to-speech model
    pub const PLAYAI_TTS_ARABIC: &str = "playai-tts-arabic";
    /// Qwen 3 32B - Alibaba's Qwen model
    pub const QWEN3_32B: &str = "qwen/qwen3-32b";

    /// All preview models
    pub const ALL: &[&str] = &[
        DEEPSEEK_R1_DISTILL_LLAMA_70B,
        LLAMA_4_MAVERICK_17B_128E_INSTRUCT,
        LLAMA_4_SCOUT_17B_16E_INSTRUCT,
        LLAMA_PROMPT_GUARD_2_22M,
        LLAMA_PROMPT_GUARD_2_86M,
        KIMI_K2_INSTRUCT,
        GPT_OSS_120B,
        GPT_OSS_20B,
        PLAYAI_TTS,
        PLAYAI_TTS_ARABIC,
        QWEN3_32B,
    ];
}

/// System models - compound systems with built-in tools
pub mod systems {
    /// Compound Beta - System with web search and code execution
    pub const COMPOUND_BETA: &str = "compound-beta";
    /// Compound Beta Mini - Smaller compound system
    pub const COMPOUND_BETA_MINI: &str = "compound-beta-mini";

    /// All system models
    pub const ALL: &[&str] = &[COMPOUND_BETA, COMPOUND_BETA_MINI];
}

/// Audio models grouped by capability
pub mod audio {
    /// Speech-to-text models
    pub mod speech_to_text {
        /// Whisper Large v3 - High-quality speech recognition
        pub const WHISPER_LARGE_V3: &str = "whisper-large-v3";
        /// Whisper Large v3 Turbo - Faster speech recognition
        pub const WHISPER_LARGE_V3_TURBO: &str = "whisper-large-v3-turbo";

        /// All speech-to-text models
        pub const ALL: &[&str] = &[WHISPER_LARGE_V3, WHISPER_LARGE_V3_TURBO];
    }

    /// Text-to-speech models
    pub mod text_to_speech {
        /// PlayAI TTS - General text-to-speech
        pub const PLAYAI_TTS: &str = "playai-tts";
        /// PlayAI TTS Arabic - Arabic text-to-speech
        pub const PLAYAI_TTS_ARABIC: &str = "playai-tts-arabic";

        /// All text-to-speech models
        pub const ALL: &[&str] = &[PLAYAI_TTS, PLAYAI_TTS_ARABIC];
    }
}

/// Chat models grouped by capability and size
pub mod chat {
    /// Large models (70B+ parameters)
    pub mod large {
        /// Llama 3.3 70B Versatile - Best balance of capability and speed
        pub const LLAMA_3_3_70B_VERSATILE: &str = "llama-3.3-70b-versatile";
        /// DeepSeek R1 Distill Llama 70B - Reasoning-focused model
        pub const DEEPSEEK_R1_DISTILL_LLAMA_70B: &str = "deepseek-r1-distill-llama-70b";
        /// OpenAI GPT-OSS 120B - Large open-source model
        pub const GPT_OSS_120B: &str = "openai/gpt-oss-120b";

        /// All large models
        pub const ALL: &[&str] = &[
            LLAMA_3_3_70B_VERSATILE,
            DEEPSEEK_R1_DISTILL_LLAMA_70B,
            GPT_OSS_120B,
        ];
    }

    /// Medium models (10B-50B parameters)
    pub mod medium {
        /// Qwen 3 32B - Alibaba's capable model
        pub const QWEN3_32B: &str = "qwen/qwen3-32b";
        /// OpenAI GPT-OSS 20B - Compact open-source model
        pub const GPT_OSS_20B: &str = "openai/gpt-oss-20b";

        /// All medium models
        pub const ALL: &[&str] = &[QWEN3_32B, GPT_OSS_20B];
    }

    /// Small models (under 10B parameters)
    pub mod small {
        /// Llama 3.1 8B Instant - Fast and efficient
        pub const LLAMA_3_1_8B_INSTANT: &str = "llama-3.1-8b-instant";
        /// Llama 4 Maverick 17B - Experimental Llama 4
        pub const LLAMA_4_MAVERICK_17B: &str = "meta-llama/llama-4-maverick-17b-128e-instruct";
        /// Llama 4 Scout 17B - Experimental Llama 4
        pub const LLAMA_4_SCOUT_17B: &str = "meta-llama/llama-4-scout-17b-16e-instruct";

        /// All small models
        pub const ALL: &[&str] = &[
            LLAMA_3_1_8B_INSTANT,
            LLAMA_4_MAVERICK_17B,
            LLAMA_4_SCOUT_17B,
        ];
    }
}

/// Popular model recommendations
pub mod popular {
    use super::*;

    /// Most capable model for general use
    pub const FLAGSHIP: &str = production::LLAMA_3_3_70B_VERSATILE;
    /// Best balance of capability and speed
    pub const BALANCED: &str = production::LLAMA_3_3_70B_VERSATILE;
    /// Fastest model for quick responses
    pub const FAST: &str = production::LLAMA_3_1_8B_INSTANT;
    /// Lightweight model for simple tasks
    pub const LIGHTWEIGHT: &str = production::LLAMA_3_1_8B_INSTANT;
    /// Best for reasoning tasks
    pub const REASONING: &str = preview::DEEPSEEK_R1_DISTILL_LLAMA_70B;
    /// Latest and most advanced
    pub const LATEST: &str = production::LLAMA_3_3_70B_VERSATILE;
    /// Best for speech-to-text
    pub const SPEECH_TO_TEXT: &str = production::WHISPER_LARGE_V3;
    /// Best for text-to-speech
    pub const TEXT_TO_SPEECH: &str = preview::PLAYAI_TTS;
    /// Best system with tools
    pub const SYSTEM: &str = systems::COMPOUND_BETA;
}

pub use preview::DEEPSEEK_R1_DISTILL_LLAMA_70B;
pub use preview::GPT_OSS_20B;
pub use preview::GPT_OSS_120B;
/// Simplified access to popular models (top-level constants)
pub use production::LLAMA_3_1_8B_INSTANT;
pub use production::LLAMA_3_3_70B_VERSATILE;
pub use production::WHISPER_LARGE_V3;
pub use systems::COMPOUND_BETA;

/// Get all available models
pub fn all_models() -> Vec<&'static str> {
    let mut models = Vec::new();
    models.extend_from_slice(production::ALL);
    models.extend_from_slice(preview::ALL);
    models.extend_from_slice(systems::ALL);
    models
}

/// Get models by capability
pub mod by_capability {
    use super::*;

    /// Models that support reasoning
    pub const REASONING: &[&str] = &[preview::DEEPSEEK_R1_DISTILL_LLAMA_70B, preview::QWEN3_32B];

    /// Models that support function calling
    pub const FUNCTION_CALLING: &[&str] = &[
        production::LLAMA_3_1_8B_INSTANT,
        production::LLAMA_3_3_70B_VERSATILE,
        preview::DEEPSEEK_R1_DISTILL_LLAMA_70B,
        preview::GPT_OSS_120B,
        preview::GPT_OSS_20B,
        preview::QWEN3_32B,
    ];

    /// Models that support audio processing
    pub const AUDIO: &[&str] = &[
        production::WHISPER_LARGE_V3,
        production::WHISPER_LARGE_V3_TURBO,
        preview::PLAYAI_TTS,
        preview::PLAYAI_TTS_ARABIC,
    ];

    /// Models optimized for speed
    pub const FAST: &[&str] = &[
        production::LLAMA_3_1_8B_INSTANT,
        production::WHISPER_LARGE_V3_TURBO,
    ];

    /// Models with built-in tools
    pub const TOOLS: &[&str] = systems::ALL;

    /// Content moderation models
    pub const MODERATION: &[&str] = &[
        production::LLAMA_GUARD_4_12B,
        preview::LLAMA_PROMPT_GUARD_2_22M,
        preview::LLAMA_PROMPT_GUARD_2_86M,
    ];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_constants() {
        // Test that constants are not empty
        assert_eq!(production::LLAMA_3_1_8B_INSTANT, "llama-3.1-8b-instant");
        assert_eq!(
            production::LLAMA_3_3_70B_VERSATILE,
            "llama-3.3-70b-versatile"
        );
        assert_eq!(
            preview::DEEPSEEK_R1_DISTILL_LLAMA_70B,
            "deepseek-r1-distill-llama-70b"
        );
        assert_eq!(systems::COMPOUND_BETA, "compound-beta");
    }

    #[test]
    fn test_all_models() {
        let models = all_models();
        assert!(!models.is_empty());
        assert!(models.contains(&production::LLAMA_3_3_70B_VERSATILE));
        assert!(models.contains(&preview::DEEPSEEK_R1_DISTILL_LLAMA_70B));
        assert!(models.contains(&systems::COMPOUND_BETA));
    }

    #[test]
    #[allow(clippy::const_is_empty)]
    fn test_popular_recommendations() {
        assert!(!popular::FLAGSHIP.is_empty());
        assert!(!popular::BALANCED.is_empty());
        assert!(!popular::REASONING.is_empty());
        assert!(!popular::FAST.is_empty());
    }

    #[test]
    #[allow(clippy::const_is_empty)]
    fn test_capability_groups() {
        assert!(!by_capability::REASONING.is_empty());
        assert!(!by_capability::FUNCTION_CALLING.is_empty());
        assert!(!by_capability::AUDIO.is_empty());
        assert!(!by_capability::FAST.is_empty());
        assert!(!by_capability::TOOLS.is_empty());
        assert!(!by_capability::MODERATION.is_empty());
    }

    #[test]
    #[allow(clippy::const_is_empty)]
    fn test_audio_models() {
        assert!(!audio::speech_to_text::ALL.is_empty());
        assert!(!audio::text_to_speech::ALL.is_empty());
        assert!(audio::speech_to_text::ALL.contains(&production::WHISPER_LARGE_V3));
        assert!(audio::text_to_speech::ALL.contains(&preview::PLAYAI_TTS));
    }

    #[test]
    #[allow(clippy::const_is_empty)]
    fn test_chat_models_by_size() {
        assert!(!chat::large::ALL.is_empty());
        assert!(!chat::medium::ALL.is_empty());
        assert!(!chat::small::ALL.is_empty());

        assert!(chat::large::ALL.contains(&production::LLAMA_3_3_70B_VERSATILE));
        assert!(chat::small::ALL.contains(&production::LLAMA_3_1_8B_INSTANT));
    }

    #[test]
    #[allow(clippy::const_is_empty)]
    fn test_model_categories() {
        assert!(!production::ALL.is_empty());
        assert!(!preview::ALL.is_empty());
        assert!(!systems::ALL.is_empty());

        // Ensure no duplicates between categories
        let production_set: std::collections::HashSet<_> = production::ALL.iter().collect();
        let preview_set: std::collections::HashSet<_> = preview::ALL.iter().collect();
        let systems_set: std::collections::HashSet<_> = systems::ALL.iter().collect();

        assert!(production_set.is_disjoint(&preview_set));
        assert!(production_set.is_disjoint(&systems_set));
        assert!(preview_set.is_disjoint(&systems_set));
    }
}
