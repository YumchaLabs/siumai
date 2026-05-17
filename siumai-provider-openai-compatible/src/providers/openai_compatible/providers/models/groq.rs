//! Groq model constants aligned with the audited AI SDK package subset.
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

pub const CHAT: &str = production::LLAMA_3_3_70B_VERSATILE;
pub const TRANSCRIPTION: &str = transcription::WHISPER_LARGE_V3_TURBO;
pub const SPEECH: &str = speech::PLAYAI_TTS;

pub const ALL_CHAT: &[&str] = chat::ALL;
pub const ALL_TRANSCRIPTION: &[&str] = transcription::ALL;
pub const ALL_SPEECH: &[&str] = speech::ALL;

/// Compatibility aliases for older Groq model imports.
pub const LLAMA_3_1_70B: &str = "llama-3.1-70b-versatile";
pub const LLAMA_3_1_8B: &str = production::LLAMA_3_1_8B_INSTANT;
pub const MIXTRAL_8X7B: &str = preview::MIXTRAL_8X7B_32768;

pub mod popular {
    use super::*;

    pub const FLAGSHIP: &str = production::LLAMA_3_3_70B_VERSATILE;
    pub const FAST: &str = production::LLAMA_3_1_8B_INSTANT;
    pub const REASONING: &str = preview::DEEPSEEK_R1_DISTILL_LLAMA_70B;
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

pub fn all_models() -> Vec<String> {
    let mut models =
        Vec::with_capacity(ALL_CHAT.len() + ALL_TRANSCRIPTION.len() + ALL_SPEECH.len());
    models.extend(ALL_CHAT.iter().map(|&model| model.to_string()));
    models.extend(ALL_TRANSCRIPTION.iter().map(|&model| model.to_string()));
    models.extend(ALL_SPEECH.iter().map(|&model| model.to_string()));
    models
}
