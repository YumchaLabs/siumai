//! OpenAI Model Constants
//!
//! This module provides convenient constants for OpenAI models, making it easy
//! for developers to reference specific models without hardcoding strings.

/// GPT-4o model family constants
pub mod gpt_4o {
    /// GPT-4o - Most capable multimodal model
    pub const GPT_4O: &str = "gpt-4o";
    /// GPT-4o dated snapshot (2024-05-13)
    pub const GPT_4O_2024_05_13: &str = "gpt-4o-2024-05-13";
    /// GPT-4o dated snapshot (2024-08-06)
    pub const GPT_4O_2024_08_06: &str = "gpt-4o-2024-08-06";
    /// GPT-4o dated snapshot (2024-11-20)
    pub const GPT_4O_2024_11_20: &str = "gpt-4o-2024-11-20";
    /// GPT-4o Mini - Fast and cost-effective multimodal model
    pub const GPT_4O_MINI: &str = "gpt-4o-mini";
    /// GPT-4o Mini dated snapshot (2024-07-18)
    pub const GPT_4O_MINI_2024_07_18: &str = "gpt-4o-mini-2024-07-18";
    /// GPT-4o Audio Preview - Latest audio-capable model
    pub const GPT_4O_AUDIO_PREVIEW: &str = "gpt-4o-audio-preview";
    /// GPT-4o Audio Preview (2024-12-17)
    pub const GPT_4O_AUDIO_PREVIEW_2024_12_17: &str = "gpt-4o-audio-preview-2024-12-17";
    /// GPT-4o Audio Preview (2024-10-01)
    pub const GPT_4O_AUDIO_PREVIEW_2024_10_01: &str = "gpt-4o-audio-preview-2024-10-01";
    /// GPT-4o Audio Preview (2025-06-03)
    pub const GPT_4O_AUDIO_PREVIEW_2025_06_03: &str = "gpt-4o-audio-preview-2025-06-03";
    /// GPT-4o Mini Audio Preview
    pub const GPT_4O_MINI_AUDIO_PREVIEW: &str = "gpt-4o-mini-audio-preview";
    /// GPT-4o Mini Audio Preview (2024-12-17)
    pub const GPT_4O_MINI_AUDIO_PREVIEW_2024_12_17: &str = "gpt-4o-mini-audio-preview-2024-12-17";
    /// GPT-4o Search Preview
    pub const GPT_4O_SEARCH_PREVIEW: &str = "gpt-4o-search-preview";
    /// GPT-4o Search Preview (2025-03-11)
    pub const GPT_4O_SEARCH_PREVIEW_2025_03_11: &str = "gpt-4o-search-preview-2025-03-11";
    /// GPT-4o Mini Search Preview
    pub const GPT_4O_MINI_SEARCH_PREVIEW: &str = "gpt-4o-mini-search-preview";
    /// GPT-4o Mini Search Preview (2025-03-11)
    pub const GPT_4O_MINI_SEARCH_PREVIEW_2025_03_11: &str = "gpt-4o-mini-search-preview-2025-03-11";

    /// All GPT-4o models
    pub const ALL: &[&str] = &[
        GPT_4O,
        GPT_4O_2024_05_13,
        GPT_4O_2024_08_06,
        GPT_4O_2024_11_20,
        GPT_4O_MINI,
        GPT_4O_MINI_2024_07_18,
        GPT_4O_AUDIO_PREVIEW,
        GPT_4O_AUDIO_PREVIEW_2024_12_17,
        GPT_4O_AUDIO_PREVIEW_2024_10_01,
        GPT_4O_AUDIO_PREVIEW_2025_06_03,
        GPT_4O_MINI_AUDIO_PREVIEW,
        GPT_4O_MINI_AUDIO_PREVIEW_2024_12_17,
        GPT_4O_SEARCH_PREVIEW,
        GPT_4O_SEARCH_PREVIEW_2025_03_11,
        GPT_4O_MINI_SEARCH_PREVIEW,
        GPT_4O_MINI_SEARCH_PREVIEW_2025_03_11,
    ];
}

/// GPT-4o mini search preview (specialized)
pub mod gpt_4o_mini_search {
    /// GPT-4o mini search preview
    pub const GPT_4O_MINI_SEARCH_PREVIEW: &str = super::gpt_4o::GPT_4O_MINI_SEARCH_PREVIEW;
    /// GPT-4o mini search preview (2025-03-11)
    pub const GPT_4O_MINI_SEARCH_PREVIEW_2025_03_11: &str =
        super::gpt_4o::GPT_4O_MINI_SEARCH_PREVIEW_2025_03_11;

    /// All GPT-4o mini search models
    pub const ALL: &[&str] = &[
        GPT_4O_MINI_SEARCH_PREVIEW,
        GPT_4O_MINI_SEARCH_PREVIEW_2025_03_11,
    ];
}

/// GPT-4.1 model family constants (new generation)
pub mod gpt_4_1 {
    /// GPT-4.1 - Next generation flagship model
    pub const GPT_4_1: &str = "gpt-4.1";
    /// GPT-4.1 dated snapshot (2025-04-14)
    pub const GPT_4_1_2025_04_14: &str = "gpt-4.1-2025-04-14";
    /// GPT-4.1 Mini - Efficient next-gen model
    pub const GPT_4_1_MINI: &str = "gpt-4.1-mini";
    /// GPT-4.1 Mini dated snapshot (2025-04-14)
    pub const GPT_4_1_MINI_2025_04_14: &str = "gpt-4.1-mini-2025-04-14";
    /// GPT-4.1 Nano - Ultra-efficient model
    pub const GPT_4_1_NANO: &str = "gpt-4.1-nano";
    /// GPT-4.1 Nano dated snapshot (2025-04-14)
    pub const GPT_4_1_NANO_2025_04_14: &str = "gpt-4.1-nano-2025-04-14";

    /// All GPT-4.1 models
    pub const ALL: &[&str] = &[
        GPT_4_1,
        GPT_4_1_2025_04_14,
        GPT_4_1_MINI,
        GPT_4_1_MINI_2025_04_14,
        GPT_4_1_NANO,
        GPT_4_1_NANO_2025_04_14,
    ];
}

/// GPT-4.5 model family constants (preview)
pub mod gpt_4_5 {
    /// GPT-4.5 Preview (2025-02-27)
    pub const GPT_4_5_PREVIEW_2025_02_27: &str = "gpt-4.5-preview-2025-02-27";
    /// GPT-4.5 Preview - Latest preview
    pub const GPT_4_5_PREVIEW: &str = "gpt-4.5-preview";
    /// GPT-4.5 - Stable release
    pub const GPT_4_5: &str = "gpt-4.5";

    /// All GPT-4.5 models
    pub const ALL: &[&str] = &[GPT_4_5_PREVIEW_2025_02_27, GPT_4_5_PREVIEW, GPT_4_5];
}

/// GPT-4 Turbo model family constants
pub mod gpt_4_turbo {
    /// GPT-4 Turbo - Latest turbo model
    pub const GPT_4_TURBO: &str = "gpt-4-turbo";
    /// GPT-4 Turbo Preview
    pub const GPT_4_TURBO_PREVIEW: &str = "gpt-4-turbo-preview";
    /// GPT-4 Turbo (2024-04-09)
    pub const GPT_4_TURBO_2024_04_09: &str = "gpt-4-turbo-2024-04-09";
    /// GPT-4 (1106 Preview)
    pub const GPT_4_1106_PREVIEW: &str = "gpt-4-1106-preview";
    /// GPT-4 (0125 Preview)
    pub const GPT_4_0125_PREVIEW: &str = "gpt-4-0125-preview";

    /// All GPT-4 Turbo models
    pub const ALL: &[&str] = &[
        GPT_4_TURBO,
        GPT_4_TURBO_PREVIEW,
        GPT_4_TURBO_2024_04_09,
        GPT_4_1106_PREVIEW,
        GPT_4_0125_PREVIEW,
    ];
}

/// GPT-4 classic model family constants
pub mod gpt_4 {
    /// GPT-4 - Original GPT-4 model
    pub const GPT_4: &str = "gpt-4";
    /// GPT-4 32K - Extended context version
    pub const GPT_4_32K: &str = "gpt-4-32k";

    /// All GPT-4 classic models
    pub const ALL: &[&str] = &[GPT_4, GPT_4_32K];
}

/// o1 reasoning model family constants
pub mod o1 {
    /// o1 - Latest reasoning model
    pub const O1: &str = "o1";
    /// o1 (2024-12-17)
    pub const O1_2024_12_17: &str = "o1-2024-12-17";
    /// o1 Preview - Preview reasoning model
    pub const O1_PREVIEW: &str = "o1-preview";
    /// o1 Mini - Efficient reasoning model
    pub const O1_MINI: &str = "o1-mini";

    /// All o1 models
    pub const ALL: &[&str] = &[O1, O1_2024_12_17, O1_PREVIEW, O1_MINI];
}

/// o3 reasoning model family constants (new)
pub mod o3 {
    /// o3 Mini - Efficient next-gen reasoning model
    pub const O3_MINI: &str = "o3-mini";
    /// o3 Mini (2025-01-31)
    pub const O3_MINI_2025_01_31: &str = "o3-mini-2025-01-31";
    /// o3 - Advanced reasoning model
    pub const O3: &str = "o3";
    /// o3 (2025-04-16)
    pub const O3_2025_04_16: &str = "o3-2025-04-16";

    /// All o3 models
    pub const ALL: &[&str] = &[O3_MINI, O3_MINI_2025_01_31, O3, O3_2025_04_16];
}

/// o4 reasoning model family constants (new)
pub mod o4 {
    /// o4 Mini - Latest efficient reasoning model
    pub const O4_MINI: &str = "o4-mini";
    /// o4 Mini (2025-04-16)
    pub const O4_MINI_2025_04_16: &str = "o4-mini-2025-04-16";

    /// All o4 models
    pub const ALL: &[&str] = &[O4_MINI, O4_MINI_2025_04_16];
}

/// GPT-5 model family constants (new generation)
pub mod gpt_5 {
    /// GPT-5 - Next generation flagship model
    pub const GPT_5: &str = "gpt-5";
    /// GPT-5 Mini - Efficient next-gen model
    pub const GPT_5_MINI: &str = "gpt-5-mini";
    /// GPT-5 Nano - Ultra-efficient model
    pub const GPT_5_NANO: &str = "gpt-5-nano";
    /// GPT-5 (2025-08-07)
    pub const GPT_5_2025_08_07: &str = "gpt-5-2025-08-07";
    /// GPT-5 Mini (2025-08-07)
    pub const GPT_5_MINI_2025_08_07: &str = "gpt-5-mini-2025-08-07";
    /// GPT-5 Nano (2025-08-07)
    pub const GPT_5_NANO_2025_08_07: &str = "gpt-5-nano-2025-08-07";
    /// GPT-5 Chat Latest - chat-optimized model, not a reasoning model.
    pub const GPT_5_CHAT_LATEST: &str = "gpt-5-chat-latest";
    /// GPT-5 Codex
    pub const GPT_5_CODEX: &str = "gpt-5-codex";
    /// GPT-5 Pro
    pub const GPT_5_PRO: &str = "gpt-5-pro";
    /// GPT-5 Pro (2025-10-06)
    pub const GPT_5_PRO_2025_10_06: &str = "gpt-5-pro-2025-10-06";

    /// GPT-5.1
    pub const GPT_5_1: &str = "gpt-5.1";
    /// GPT-5.1 (2025-11-13)
    pub const GPT_5_1_2025_11_13: &str = "gpt-5.1-2025-11-13";
    /// GPT-5.1 Chat Latest
    pub const GPT_5_1_CHAT_LATEST: &str = "gpt-5.1-chat-latest";
    /// GPT-5.1 Codex Mini
    pub const GPT_5_1_CODEX_MINI: &str = "gpt-5.1-codex-mini";
    /// GPT-5.1 Codex
    pub const GPT_5_1_CODEX: &str = "gpt-5.1-codex";
    /// GPT-5.1 Codex Max
    pub const GPT_5_1_CODEX_MAX: &str = "gpt-5.1-codex-max";

    /// GPT-5.2
    pub const GPT_5_2: &str = "gpt-5.2";
    /// GPT-5.2 (2025-12-11)
    pub const GPT_5_2_2025_12_11: &str = "gpt-5.2-2025-12-11";
    /// GPT-5.2 Chat Latest
    pub const GPT_5_2_CHAT_LATEST: &str = "gpt-5.2-chat-latest";
    /// GPT-5.2 Pro
    pub const GPT_5_2_PRO: &str = "gpt-5.2-pro";
    /// GPT-5.2 Pro (2025-12-11)
    pub const GPT_5_2_PRO_2025_12_11: &str = "gpt-5.2-pro-2025-12-11";
    /// GPT-5.2 Codex
    pub const GPT_5_2_CODEX: &str = "gpt-5.2-codex";

    /// GPT-5.3 Chat Latest
    pub const GPT_5_3_CHAT_LATEST: &str = "gpt-5.3-chat-latest";
    /// GPT-5.3 Codex
    pub const GPT_5_3_CODEX: &str = "gpt-5.3-codex";

    /// GPT-5.4
    pub const GPT_5_4: &str = "gpt-5.4";
    /// GPT-5.4 (2026-03-05)
    pub const GPT_5_4_2026_03_05: &str = "gpt-5.4-2026-03-05";
    /// GPT-5.4 Mini
    pub const GPT_5_4_MINI: &str = "gpt-5.4-mini";
    /// GPT-5.4 Mini (2026-03-17)
    pub const GPT_5_4_MINI_2026_03_17: &str = "gpt-5.4-mini-2026-03-17";
    /// GPT-5.4 Nano
    pub const GPT_5_4_NANO: &str = "gpt-5.4-nano";
    /// GPT-5.4 Nano (2026-03-17)
    pub const GPT_5_4_NANO_2026_03_17: &str = "gpt-5.4-nano-2026-03-17";
    /// GPT-5.4 Pro
    pub const GPT_5_4_PRO: &str = "gpt-5.4-pro";
    /// GPT-5.4 Pro (2026-03-05)
    pub const GPT_5_4_PRO_2026_03_05: &str = "gpt-5.4-pro-2026-03-05";

    /// GPT-5.5
    pub const GPT_5_5: &str = "gpt-5.5";
    /// GPT-5.5 (2026-04-23)
    pub const GPT_5_5_2026_04_23: &str = "gpt-5.5-2026-04-23";

    /// All GPT-5 models
    pub const ALL: &[&str] = &[
        GPT_5,
        GPT_5_MINI,
        GPT_5_NANO,
        GPT_5_2025_08_07,
        GPT_5_MINI_2025_08_07,
        GPT_5_NANO_2025_08_07,
        GPT_5_CHAT_LATEST,
        GPT_5_CODEX,
        GPT_5_PRO,
        GPT_5_PRO_2025_10_06,
        GPT_5_1,
        GPT_5_1_2025_11_13,
        GPT_5_1_CHAT_LATEST,
        GPT_5_1_CODEX_MINI,
        GPT_5_1_CODEX,
        GPT_5_1_CODEX_MAX,
        GPT_5_2,
        GPT_5_2_2025_12_11,
        GPT_5_2_CHAT_LATEST,
        GPT_5_2_PRO,
        GPT_5_2_PRO_2025_12_11,
        GPT_5_2_CODEX,
        GPT_5_3_CHAT_LATEST,
        GPT_5_3_CODEX,
        GPT_5_4,
        GPT_5_4_2026_03_05,
        GPT_5_4_MINI,
        GPT_5_4_MINI_2026_03_17,
        GPT_5_4_NANO,
        GPT_5_4_NANO_2026_03_17,
        GPT_5_4_PRO,
        GPT_5_4_PRO_2026_03_05,
        GPT_5_5,
        GPT_5_5_2026_04_23,
    ];

    /// GPT-5 models that are treated as reasoning models by Vercel AI SDK.
    pub const REASONING: &[&str] = &[
        GPT_5,
        GPT_5_2025_08_07,
        GPT_5_CODEX,
        GPT_5_MINI,
        GPT_5_MINI_2025_08_07,
        GPT_5_NANO,
        GPT_5_NANO_2025_08_07,
        GPT_5_PRO,
        GPT_5_PRO_2025_10_06,
        GPT_5_1,
        GPT_5_1_2025_11_13,
        GPT_5_1_CHAT_LATEST,
        GPT_5_1_CODEX_MINI,
        GPT_5_1_CODEX,
        GPT_5_1_CODEX_MAX,
        GPT_5_2,
        GPT_5_2_2025_12_11,
        GPT_5_2_CHAT_LATEST,
        GPT_5_2_PRO,
        GPT_5_2_PRO_2025_12_11,
        GPT_5_2_CODEX,
        GPT_5_3_CHAT_LATEST,
        GPT_5_3_CODEX,
        GPT_5_4,
        GPT_5_4_2026_03_05,
        GPT_5_4_MINI,
        GPT_5_4_MINI_2026_03_17,
        GPT_5_4_NANO,
        GPT_5_4_NANO_2026_03_17,
        GPT_5_4_PRO,
        GPT_5_4_PRO_2026_03_05,
        GPT_5_5,
        GPT_5_5_2026_04_23,
    ];
}

/// GPT-3.5 model family constants
pub mod gpt_3_5 {
    /// GPT-3.5 Turbo - Most capable GPT-3.5 model
    pub const GPT_3_5_TURBO: &str = "gpt-3.5-turbo";
    /// GPT-3.5 Turbo (0125)
    pub const GPT_3_5_TURBO_0125: &str = "gpt-3.5-turbo-0125";
    /// GPT-3.5 Turbo (1106)
    pub const GPT_3_5_TURBO_1106: &str = "gpt-3.5-turbo-1106";
    /// GPT-3.5 Turbo 16K - Extended context version
    pub const GPT_3_5_TURBO_16K: &str = "gpt-3.5-turbo-16k";
    /// GPT-3.5 Turbo Instruct - Completion model
    pub const GPT_3_5_TURBO_INSTRUCT: &str = "gpt-3.5-turbo-instruct";
    /// GPT-3.5 Turbo Instruct (0914)
    pub const GPT_3_5_TURBO_INSTRUCT_0914: &str = "gpt-3.5-turbo-instruct-0914";

    /// All GPT-3.5 models
    pub const ALL: &[&str] = &[
        GPT_3_5_TURBO,
        GPT_3_5_TURBO_0125,
        GPT_3_5_TURBO_1106,
        GPT_3_5_TURBO_16K,
        GPT_3_5_TURBO_INSTRUCT,
        GPT_3_5_TURBO_INSTRUCT_0914,
    ];
}

/// Audio model constants
pub mod audio {
    /// TTS-1 - Text-to-speech model
    pub const TTS_1: &str = "tts-1";
    /// TTS-1 (1106)
    pub const TTS_1_1106: &str = "tts-1-1106";
    /// TTS-1 HD - High-definition text-to-speech
    pub const TTS_1_HD: &str = "tts-1-hd";
    /// TTS-1 HD (1106)
    pub const TTS_1_HD_1106: &str = "tts-1-hd-1106";
    /// GPT-4o Mini TTS
    pub const GPT_4O_MINI_TTS: &str = "gpt-4o-mini-tts";
    /// GPT-4o Mini TTS (2025-03-20)
    pub const GPT_4O_MINI_TTS_2025_03_20: &str = "gpt-4o-mini-tts-2025-03-20";
    /// GPT-4o Mini TTS (2025-12-15)
    pub const GPT_4O_MINI_TTS_2025_12_15: &str = "gpt-4o-mini-tts-2025-12-15";
    /// Whisper-1 - Speech-to-text model
    pub const WHISPER_1: &str = "whisper-1";
    /// GPT-4o Mini Transcribe
    pub const GPT_4O_MINI_TRANSCRIBE: &str = "gpt-4o-mini-transcribe";
    /// GPT-4o Mini Transcribe (2025-03-20)
    pub const GPT_4O_MINI_TRANSCRIBE_2025_03_20: &str = "gpt-4o-mini-transcribe-2025-03-20";
    /// GPT-4o Mini Transcribe (2025-12-15)
    pub const GPT_4O_MINI_TRANSCRIBE_2025_12_15: &str = "gpt-4o-mini-transcribe-2025-12-15";
    /// GPT-4o Transcribe
    pub const GPT_4O_TRANSCRIBE: &str = "gpt-4o-transcribe";
    /// GPT-4o Transcribe Diarize
    pub const GPT_4O_TRANSCRIBE_DIARIZE: &str = "gpt-4o-transcribe-diarize";

    /// All audio models
    pub const ALL: &[&str] = &[
        TTS_1,
        TTS_1_1106,
        TTS_1_HD,
        TTS_1_HD_1106,
        GPT_4O_MINI_TTS,
        GPT_4O_MINI_TTS_2025_03_20,
        GPT_4O_MINI_TTS_2025_12_15,
        WHISPER_1,
        GPT_4O_MINI_TRANSCRIBE,
        GPT_4O_MINI_TRANSCRIBE_2025_03_20,
        GPT_4O_MINI_TRANSCRIBE_2025_12_15,
        GPT_4O_TRANSCRIBE,
        GPT_4O_TRANSCRIBE_DIARIZE,
    ];

    /// Text-to-speech models.
    pub const SPEECH: &[&str] = &[
        TTS_1,
        TTS_1_1106,
        TTS_1_HD,
        TTS_1_HD_1106,
        GPT_4O_MINI_TTS,
        GPT_4O_MINI_TTS_2025_03_20,
        GPT_4O_MINI_TTS_2025_12_15,
    ];

    /// Speech-to-text transcription models.
    pub const TRANSCRIPTION: &[&str] = &[
        WHISPER_1,
        GPT_4O_MINI_TRANSCRIBE,
        GPT_4O_MINI_TRANSCRIBE_2025_03_20,
        GPT_4O_MINI_TRANSCRIBE_2025_12_15,
        GPT_4O_TRANSCRIBE,
        GPT_4O_TRANSCRIBE_DIARIZE,
    ];
}

/// Image generation model constants
pub mod images {
    /// DALL-E 2 - Image generation model
    pub const DALL_E_2: &str = "dall-e-2";
    /// DALL-E 3 - Advanced image generation model
    pub const DALL_E_3: &str = "dall-e-3";
    /// GPT Image 1 - Multimodal image generation and editing model
    pub const GPT_IMAGE_1: &str = "gpt-image-1";
    /// GPT Image 1 Mini - Lower-cost GPT image model
    pub const GPT_IMAGE_1_MINI: &str = "gpt-image-1-mini";
    /// GPT Image 1.5 - GPT image model
    pub const GPT_IMAGE_1_5: &str = "gpt-image-1.5";
    /// GPT Image 2 - Latest GPT image generation and editing model
    pub const GPT_IMAGE_2: &str = "gpt-image-2";
    /// ChatGPT Image Latest - ChatGPT image model alias
    pub const CHATGPT_IMAGE_LATEST: &str = "chatgpt-image-latest";

    /// All image models
    pub const ALL: &[&str] = &[
        DALL_E_2,
        DALL_E_3,
        GPT_IMAGE_1,
        GPT_IMAGE_1_MINI,
        GPT_IMAGE_1_5,
        GPT_IMAGE_2,
        CHATGPT_IMAGE_LATEST,
    ];
}

/// Embedding model constants
pub mod embeddings {
    /// Text Embedding 3 Small - Efficient embedding model
    pub const TEXT_EMBEDDING_3_SMALL: &str = "text-embedding-3-small";
    /// Text Embedding 3 Large - High-performance embedding model
    pub const TEXT_EMBEDDING_3_LARGE: &str = "text-embedding-3-large";
    /// Text Embedding Ada 002 - Legacy embedding model
    pub const TEXT_EMBEDDING_ADA_002: &str = "text-embedding-ada-002";

    /// All embedding models
    pub const ALL: &[&str] = &[
        TEXT_EMBEDDING_3_SMALL,
        TEXT_EMBEDDING_3_LARGE,
        TEXT_EMBEDDING_ADA_002,
    ];
}

/// Moderation model constants
pub mod moderation {
    /// Omni Moderation Latest - latest multimodal moderation model.
    pub const OMNI_MODERATION_LATEST: &str = "omni-moderation-latest";
    /// Text Moderation Latest - Latest moderation model
    pub const TEXT_MODERATION_LATEST: &str = "text-moderation-latest";
    /// Text Moderation Stable - Stable moderation model
    pub const TEXT_MODERATION_STABLE: &str = "text-moderation-stable";

    /// All moderation models
    pub const ALL: &[&str] = &[
        OMNI_MODERATION_LATEST,
        TEXT_MODERATION_LATEST,
        TEXT_MODERATION_STABLE,
    ];
}

/// Popular model recommendations
pub mod popular {
    use super::*;

    /// Most capable model for general use
    pub const FLAGSHIP: &str = gpt_4o::GPT_4O;
    /// Best balance of capability and cost
    pub const BALANCED: &str = gpt_4o::GPT_4O_MINI;
    /// Best for reasoning tasks
    pub const REASONING: &str = o1::O1;
    /// Most cost-effective for simple tasks
    pub const ECONOMICAL: &str = gpt_3_5::GPT_3_5_TURBO;
    /// Latest and most advanced
    pub const LATEST: &str = gpt_5::GPT_5;
}

/// Get all chat models
pub fn all_chat_models() -> Vec<&'static str> {
    let mut models = Vec::new();
    models.extend_from_slice(gpt_4o::ALL);
    models.extend_from_slice(gpt_4o_mini_search::ALL);
    models.extend_from_slice(gpt_4_1::ALL);
    models.extend_from_slice(gpt_4_5::ALL);
    models.extend_from_slice(gpt_4_turbo::ALL);
    models.extend_from_slice(gpt_4::ALL);
    models.extend_from_slice(o1::ALL);
    models.extend_from_slice(o3::ALL);
    models.extend_from_slice(o4::ALL);
    models.extend_from_slice(gpt_5::ALL);
    models.extend_from_slice(gpt_3_5::ALL);
    models.sort_unstable();
    models.dedup();
    models
}

/// Get all reasoning models
pub fn all_reasoning_models() -> Vec<&'static str> {
    let mut models = Vec::new();
    models.extend_from_slice(o1::ALL);
    models.extend_from_slice(o3::ALL);
    models.extend_from_slice(o4::ALL);
    models.extend_from_slice(gpt_5::REASONING);
    models.sort_unstable();
    models.dedup();
    models
}

/// Get all multimodal models (vision + audio capable)
pub fn all_multimodal_models() -> Vec<&'static str> {
    let mut models = Vec::new();
    models.extend_from_slice(gpt_4o::ALL);
    models.extend_from_slice(gpt_4_1::ALL);
    models.extend_from_slice(gpt_4_5::ALL);
    models.extend_from_slice(gpt_5::ALL);
    models.sort_unstable();
    models.dedup();
    models
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_constants_include_vercel_ai_sdk_openai_surface() {
        for model_id in [
            gpt_4o::GPT_4O_2024_05_13,
            gpt_4o::GPT_4O_AUDIO_PREVIEW_2025_06_03,
            gpt_4o::GPT_4O_SEARCH_PREVIEW_2025_03_11,
            gpt_4o::GPT_4O_MINI_SEARCH_PREVIEW_2025_03_11,
            gpt_4_1::GPT_4_1_2025_04_14,
            o3::O3_2025_04_16,
            o4::O4_MINI_2025_04_16,
            gpt_5::GPT_5_CHAT_LATEST,
            gpt_5::GPT_5_CODEX,
            gpt_5::GPT_5_PRO_2025_10_06,
            gpt_5::GPT_5_1_CODEX_MAX,
            gpt_5::GPT_5_2_PRO_2025_12_11,
            gpt_5::GPT_5_3_CODEX,
            gpt_5::GPT_5_4_PRO_2026_03_05,
            gpt_5::GPT_5_5_2026_04_23,
            gpt_3_5::GPT_3_5_TURBO_0125,
            gpt_3_5::GPT_3_5_TURBO_INSTRUCT_0914,
        ] {
            assert!(
                all_chat_models().contains(&model_id),
                "missing chat model constant from catalog: {model_id}"
            );
        }

        for model_id in [
            gpt_5::GPT_5_CODEX,
            gpt_5::GPT_5_1_CODEX,
            gpt_5::GPT_5_2_CODEX,
            gpt_5::GPT_5_4_NANO,
            gpt_5::GPT_5_5,
        ] {
            assert!(
                all_reasoning_models().contains(&model_id),
                "missing reasoning model constant from catalog: {model_id}"
            );
        }

        for model_id in [
            audio::TTS_1_1106,
            audio::TTS_1_HD_1106,
            audio::GPT_4O_MINI_TTS_2025_12_15,
            audio::GPT_4O_MINI_TRANSCRIBE_2025_12_15,
            audio::GPT_4O_TRANSCRIBE_DIARIZE,
        ] {
            assert!(
                audio::ALL.contains(&model_id),
                "missing audio model constant from catalog: {model_id}"
            );
        }
    }
}
