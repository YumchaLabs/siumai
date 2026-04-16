//! Groq model constants.
//!
//! This module keeps the public Rust catalog aligned with the audited `@ai-sdk/groq` package
//! surface for chat/transcription model ids, while preserving Groq's provider-owned TTS support
//! as an explicit extension outside that upstream package contract.

/// Production chat models exposed by `@ai-sdk/groq`.
pub mod production {
    pub const GEMMA2_9B_IT: &str = "gemma2-9b-it";
    pub const LLAMA_3_1_8B_INSTANT: &str = "llama-3.1-8b-instant";
    pub const LLAMA_3_3_70B_VERSATILE: &str = "llama-3.3-70b-versatile";
    pub const LLAMA_GUARD_4_12B: &str = "meta-llama/llama-guard-4-12b";
    pub const GPT_OSS_120B: &str = "openai/gpt-oss-120b";
    pub const GPT_OSS_20B: &str = "openai/gpt-oss-20b";

    pub const ALL: &[&str] = &[
        GEMMA2_9B_IT,
        LLAMA_3_1_8B_INSTANT,
        LLAMA_3_3_70B_VERSATILE,
        LLAMA_GUARD_4_12B,
        GPT_OSS_120B,
        GPT_OSS_20B,
    ];
}

/// Preview chat models exposed by `@ai-sdk/groq`.
pub mod preview {
    pub const DEEPSEEK_R1_DISTILL_LLAMA_70B: &str = "deepseek-r1-distill-llama-70b";
    pub const LLAMA_4_MAVERICK_17B_128E_INSTRUCT: &str =
        "meta-llama/llama-4-maverick-17b-128e-instruct";
    pub const LLAMA_4_SCOUT_17B_16E_INSTRUCT: &str = "meta-llama/llama-4-scout-17b-16e-instruct";
    pub const LLAMA_PROMPT_GUARD_2_22M: &str = "meta-llama/llama-prompt-guard-2-22m";
    pub const LLAMA_PROMPT_GUARD_2_86M: &str = "meta-llama/llama-prompt-guard-2-86m";
    pub const KIMI_K2_INSTRUCT: &str = "moonshotai/kimi-k2-instruct-0905";
    pub const QWEN3_32B: &str = "qwen/qwen3-32b";
    pub const LLAMA_GUARD_3_8B: &str = "llama-guard-3-8b";
    pub const LLAMA3_70B_8192: &str = "llama3-70b-8192";
    pub const LLAMA3_8B_8192: &str = "llama3-8b-8192";
    pub const MIXTRAL_8X7B_32768: &str = "mixtral-8x7b-32768";
    pub const QWEN_QWQ_32B: &str = "qwen-qwq-32b";
    pub const QWEN_2_5_32B: &str = "qwen-2.5-32b";
    pub const DEEPSEEK_R1_DISTILL_QWEN_32B: &str = "deepseek-r1-distill-qwen-32b";

    pub const ALL: &[&str] = &[
        DEEPSEEK_R1_DISTILL_LLAMA_70B,
        LLAMA_4_MAVERICK_17B_128E_INSTRUCT,
        LLAMA_4_SCOUT_17B_16E_INSTRUCT,
        LLAMA_PROMPT_GUARD_2_22M,
        LLAMA_PROMPT_GUARD_2_86M,
        KIMI_K2_INSTRUCT,
        QWEN3_32B,
        LLAMA_GUARD_3_8B,
        LLAMA3_70B_8192,
        LLAMA3_8B_8192,
        MIXTRAL_8X7B_32768,
        QWEN_QWQ_32B,
        QWEN_2_5_32B,
        DEEPSEEK_R1_DISTILL_QWEN_32B,
    ];
}

/// Transcription models exposed by `@ai-sdk/groq`.
pub mod transcription {
    pub const WHISPER_LARGE_V3_TURBO: &str = "whisper-large-v3-turbo";
    pub const WHISPER_LARGE_V3: &str = "whisper-large-v3";

    pub const ALL: &[&str] = &[WHISPER_LARGE_V3_TURBO, WHISPER_LARGE_V3];
}

/// Provider-owned Groq speech models not part of the upstream AI SDK package surface.
pub mod speech {
    pub const PLAYAI_TTS: &str = "playai-tts";
    pub const PLAYAI_TTS_ARABIC: &str = "playai-tts-arabic";

    pub const ALL: &[&str] = &[PLAYAI_TTS, PLAYAI_TTS_ARABIC];
}

/// Compatibility aliases for older Groq model imports.
pub mod audio {
    pub mod speech_to_text {
        pub use super::super::transcription::*;
    }

    pub mod text_to_speech {
        pub use super::super::speech::*;
    }
}

/// AI SDK-aligned Groq chat catalog.
pub mod chat {
    pub mod production {
        pub use super::super::production::*;
    }

    pub mod preview {
        pub use super::super::preview::*;
    }

    pub const ALL: &[&str] = &[
        super::production::GEMMA2_9B_IT,
        super::production::LLAMA_3_1_8B_INSTANT,
        super::production::LLAMA_3_3_70B_VERSATILE,
        super::production::LLAMA_GUARD_4_12B,
        super::production::GPT_OSS_120B,
        super::production::GPT_OSS_20B,
        super::preview::DEEPSEEK_R1_DISTILL_LLAMA_70B,
        super::preview::LLAMA_4_MAVERICK_17B_128E_INSTRUCT,
        super::preview::LLAMA_4_SCOUT_17B_16E_INSTRUCT,
        super::preview::LLAMA_PROMPT_GUARD_2_22M,
        super::preview::LLAMA_PROMPT_GUARD_2_86M,
        super::preview::KIMI_K2_INSTRUCT,
        super::preview::QWEN3_32B,
        super::preview::LLAMA_GUARD_3_8B,
        super::preview::LLAMA3_70B_8192,
        super::preview::LLAMA3_8B_8192,
        super::preview::MIXTRAL_8X7B_32768,
        super::preview::QWEN_QWQ_32B,
        super::preview::QWEN_2_5_32B,
        super::preview::DEEPSEEK_R1_DISTILL_QWEN_32B,
    ];
}

pub const ALL_CHAT: &[&str] = chat::ALL;
pub const ALL_TRANSCRIPTION: &[&str] = transcription::ALL;
pub const ALL_SPEECH: &[&str] = speech::ALL;

/// Recommended model shortcuts.
pub mod popular {
    use super::*;

    pub const FLAGSHIP: &str = production::LLAMA_3_3_70B_VERSATILE;
    pub const BALANCED: &str = production::LLAMA_3_3_70B_VERSATILE;
    pub const FAST: &str = production::LLAMA_3_1_8B_INSTANT;
    pub const LIGHTWEIGHT: &str = production::LLAMA_3_1_8B_INSTANT;
    pub const REASONING: &str = preview::DEEPSEEK_R1_DISTILL_LLAMA_70B;
    pub const LATEST: &str = production::LLAMA_3_3_70B_VERSATILE;
    pub const SPEECH_TO_TEXT: &str = transcription::WHISPER_LARGE_V3;
    pub const TEXT_TO_SPEECH: &str = speech::PLAYAI_TTS;
}

pub use preview::DEEPSEEK_R1_DISTILL_LLAMA_70B;
pub use production::GPT_OSS_20B;
pub use production::GPT_OSS_120B;
pub use production::LLAMA_3_1_8B_INSTANT;
pub use production::LLAMA_3_3_70B_VERSATILE;
pub use speech::PLAYAI_TTS;
pub use transcription::WHISPER_LARGE_V3;

/// Get all Groq models exposed by the Rust provider.
pub fn all_models() -> Vec<&'static str> {
    let mut models =
        Vec::with_capacity(ALL_CHAT.len() + ALL_TRANSCRIPTION.len() + ALL_SPEECH.len());
    models.extend_from_slice(ALL_CHAT);
    models.extend_from_slice(ALL_TRANSCRIPTION);
    models.extend_from_slice(ALL_SPEECH);
    models
}

/// Capability-oriented Groq model groups.
pub mod by_capability {
    use super::*;

    pub const REASONING: &[&str] = &[
        preview::DEEPSEEK_R1_DISTILL_LLAMA_70B,
        preview::DEEPSEEK_R1_DISTILL_QWEN_32B,
        preview::QWEN3_32B,
        preview::QWEN_QWQ_32B,
    ];

    pub const FUNCTION_CALLING: &[&str] = chat::ALL;

    pub const BROWSER_SEARCH: &[&str] = &[production::GPT_OSS_20B, production::GPT_OSS_120B];

    pub const SPEECH_TO_TEXT: &[&str] = transcription::ALL;

    pub const TEXT_TO_SPEECH: &[&str] = speech::ALL;

    pub const AUDIO: &[&str] = &[
        transcription::WHISPER_LARGE_V3_TURBO,
        transcription::WHISPER_LARGE_V3,
        speech::PLAYAI_TTS,
        speech::PLAYAI_TTS_ARABIC,
    ];

    pub const FAST: &[&str] = &[
        production::LLAMA_3_1_8B_INSTANT,
        transcription::WHISPER_LARGE_V3_TURBO,
    ];

    pub const MODERATION: &[&str] = &[
        production::LLAMA_GUARD_4_12B,
        preview::LLAMA_GUARD_3_8B,
        preview::LLAMA_PROMPT_GUARD_2_22M,
        preview::LLAMA_PROMPT_GUARD_2_86M,
    ];
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXPECTED_AI_SDK_CHAT_MODELS: &[&str] = &[
        production::GEMMA2_9B_IT,
        production::LLAMA_3_1_8B_INSTANT,
        production::LLAMA_3_3_70B_VERSATILE,
        production::LLAMA_GUARD_4_12B,
        production::GPT_OSS_120B,
        production::GPT_OSS_20B,
        preview::DEEPSEEK_R1_DISTILL_LLAMA_70B,
        preview::LLAMA_4_MAVERICK_17B_128E_INSTRUCT,
        preview::LLAMA_4_SCOUT_17B_16E_INSTRUCT,
        preview::LLAMA_PROMPT_GUARD_2_22M,
        preview::LLAMA_PROMPT_GUARD_2_86M,
        preview::KIMI_K2_INSTRUCT,
        preview::QWEN3_32B,
        preview::LLAMA_GUARD_3_8B,
        preview::LLAMA3_70B_8192,
        preview::LLAMA3_8B_8192,
        preview::MIXTRAL_8X7B_32768,
        preview::QWEN_QWQ_32B,
        preview::QWEN_2_5_32B,
        preview::DEEPSEEK_R1_DISTILL_QWEN_32B,
    ];

    #[test]
    fn chat_catalog_matches_audited_ai_sdk_surface() {
        assert_eq!(chat::ALL, EXPECTED_AI_SDK_CHAT_MODELS);
    }

    #[test]
    fn transcription_catalog_matches_audited_ai_sdk_surface() {
        assert_eq!(
            transcription::ALL,
            &[
                transcription::WHISPER_LARGE_V3_TURBO,
                transcription::WHISPER_LARGE_V3
            ]
        );
    }

    #[test]
    fn provider_owned_tts_models_stay_outside_chat_catalog() {
        assert!(!chat::ALL.contains(&speech::PLAYAI_TTS));
        assert!(!chat::ALL.contains(&speech::PLAYAI_TTS_ARABIC));
    }

    #[test]
    fn obsolete_chat_models_are_removed_from_public_catalog() {
        let models = all_models();

        for obsolete in [
            "compound-beta",
            "compound-beta-mini",
            "llama-3.2-11b-vision-preview",
            "llama-3.2-90b-vision-preview",
            "llama3-groq-8b-8192-tool-use-preview",
            "llama3-groq-70b-8192-tool-use-preview",
            "llava-v1.5-7b-4096-preview",
            "gemma-7b-it",
        ] {
            assert!(
                !models.contains(&obsolete),
                "obsolete model should be removed: {obsolete}"
            );
        }
    }

    #[test]
    fn all_models_include_chat_and_audio_catalogs_without_duplicates() {
        let models = all_models();

        assert!(models.contains(&production::LLAMA_3_3_70B_VERSATILE));
        assert!(models.contains(&preview::DEEPSEEK_R1_DISTILL_QWEN_32B));
        assert!(models.contains(&transcription::WHISPER_LARGE_V3));
        assert!(models.contains(&speech::PLAYAI_TTS));

        let set: std::collections::HashSet<_> = models.iter().collect();
        assert_eq!(set.len(), models.len());
    }

    #[test]
    fn popular_recommendations_reference_live_catalog_entries() {
        for model in [
            popular::FLAGSHIP,
            popular::BALANCED,
            popular::FAST,
            popular::LIGHTWEIGHT,
            popular::REASONING,
            popular::LATEST,
            popular::SPEECH_TO_TEXT,
            popular::TEXT_TO_SPEECH,
        ] {
            assert!(
                all_models().contains(&model),
                "popular model should exist: {model}"
            );
        }
    }

    #[test]
    fn capability_groups_reference_live_catalog_entries() {
        for group in [
            by_capability::REASONING,
            by_capability::FUNCTION_CALLING,
            by_capability::BROWSER_SEARCH,
            by_capability::SPEECH_TO_TEXT,
            by_capability::TEXT_TO_SPEECH,
            by_capability::AUDIO,
            by_capability::FAST,
            by_capability::MODERATION,
        ] {
            assert!(!group.is_empty());
            for model in group {
                assert!(
                    all_models().contains(model),
                    "capability model should exist: {model}"
                );
            }
        }
    }
}
