//! OpenAI Model Constants
//!
//! This module provides convenient constants for OpenAI models, making it easy
//! for developers to reference specific models without hardcoding strings.

/// GPT-4o model family constants
pub mod gpt_4o {
    /// GPT-4o - Most capable multimodal model
    pub const GPT_4O: &str = "gpt-4o";
    /// GPT-4o Mini - Fast and cost-effective multimodal model
    pub const GPT_4O_MINI: &str = "gpt-4o-mini";
    /// GPT-4o Audio Preview - Latest audio-capable model
    pub const GPT_4O_AUDIO_PREVIEW: &str = "gpt-4o-audio-preview";
    /// GPT-4o Audio Preview (2024-12-17)
    pub const GPT_4O_AUDIO_PREVIEW_2024_12_17: &str = "gpt-4o-audio-preview-2024-12-17";
    /// GPT-4o Audio Preview (2024-10-01)
    pub const GPT_4O_AUDIO_PREVIEW_2024_10_01: &str = "gpt-4o-audio-preview-2024-10-01";
    /// GPT-4o Mini Audio Preview
    pub const GPT_4O_MINI_AUDIO_PREVIEW: &str = "gpt-4o-mini-audio-preview";
    /// GPT-4o Mini Audio Preview (2024-12-17)
    pub const GPT_4O_MINI_AUDIO_PREVIEW_2024_12_17: &str = "gpt-4o-mini-audio-preview-2024-12-17";

    /// All GPT-4o models
    pub const ALL: &[&str] = &[
        GPT_4O,
        GPT_4O_MINI,
        GPT_4O_AUDIO_PREVIEW,
        GPT_4O_AUDIO_PREVIEW_2024_12_17,
        GPT_4O_AUDIO_PREVIEW_2024_10_01,
        GPT_4O_MINI_AUDIO_PREVIEW,
        GPT_4O_MINI_AUDIO_PREVIEW_2024_12_17,
    ];
}

/// GPT-4.1 model family constants (new generation)
pub mod gpt_4_1 {
    /// GPT-4.1 - Next generation flagship model
    pub const GPT_4_1: &str = "gpt-4.1";
    /// GPT-4.1 Mini - Efficient next-gen model
    pub const GPT_4_1_MINI: &str = "gpt-4.1-mini";
    /// GPT-4.1 Nano - Ultra-efficient model
    pub const GPT_4_1_NANO: &str = "gpt-4.1-nano";

    /// All GPT-4.1 models
    pub const ALL: &[&str] = &[GPT_4_1, GPT_4_1_MINI, GPT_4_1_NANO];
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
    /// o3 - Advanced reasoning model
    pub const O3: &str = "o3";

    /// All o3 models
    pub const ALL: &[&str] = &[O3_MINI, O3];
}

/// o4 reasoning model family constants (new)
pub mod o4 {
    /// o4 Mini - Latest efficient reasoning model
    pub const O4_MINI: &str = "o4-mini";

    /// All o4 models
    pub const ALL: &[&str] = &[O4_MINI];
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

    /// All GPT-5 models
    pub const ALL: &[&str] = &[
        GPT_5,
        GPT_5_MINI,
        GPT_5_NANO,
        GPT_5_2025_08_07,
        GPT_5_MINI_2025_08_07,
        GPT_5_NANO_2025_08_07,
    ];
}

/// GPT-3.5 model family constants
pub mod gpt_3_5 {
    /// GPT-3.5 Turbo - Most capable GPT-3.5 model
    pub const GPT_3_5_TURBO: &str = "gpt-3.5-turbo";
    /// GPT-3.5 Turbo 16K - Extended context version
    pub const GPT_3_5_TURBO_16K: &str = "gpt-3.5-turbo-16k";
    /// GPT-3.5 Turbo Instruct - Completion model
    pub const GPT_3_5_TURBO_INSTRUCT: &str = "gpt-3.5-turbo-instruct";

    /// All GPT-3.5 models
    pub const ALL: &[&str] = &[GPT_3_5_TURBO, GPT_3_5_TURBO_16K, GPT_3_5_TURBO_INSTRUCT];
}

/// Audio model constants
pub mod audio {
    /// TTS-1 - Text-to-speech model
    pub const TTS_1: &str = "tts-1";
    /// TTS-1 HD - High-definition text-to-speech
    pub const TTS_1_HD: &str = "tts-1-hd";
    /// Whisper-1 - Speech-to-text model
    pub const WHISPER_1: &str = "whisper-1";

    /// All audio models
    pub const ALL: &[&str] = &[TTS_1, TTS_1_HD, WHISPER_1];
}

/// Image generation model constants
pub mod images {
    /// DALL-E 2 - Image generation model
    pub const DALL_E_2: &str = "dall-e-2";
    /// DALL-E 3 - Advanced image generation model
    pub const DALL_E_3: &str = "dall-e-3";

    /// All image models
    pub const ALL: &[&str] = &[DALL_E_2, DALL_E_3];
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
    /// Text Moderation Latest - Latest moderation model
    pub const TEXT_MODERATION_LATEST: &str = "text-moderation-latest";
    /// Text Moderation Stable - Stable moderation model
    pub const TEXT_MODERATION_STABLE: &str = "text-moderation-stable";

    /// All moderation models
    pub const ALL: &[&str] = &[TEXT_MODERATION_LATEST, TEXT_MODERATION_STABLE];
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
    models.extend_from_slice(gpt_4_1::ALL);
    models.extend_from_slice(gpt_4_5::ALL);
    models.extend_from_slice(gpt_4_turbo::ALL);
    models.extend_from_slice(gpt_4::ALL);
    models.extend_from_slice(o1::ALL);
    models.extend_from_slice(o3::ALL);
    models.extend_from_slice(o4::ALL);
    models.extend_from_slice(gpt_5::ALL);
    models.extend_from_slice(gpt_3_5::ALL);
    models
}

/// Get all reasoning models
pub fn all_reasoning_models() -> Vec<&'static str> {
    let mut models = Vec::new();
    models.extend_from_slice(o1::ALL);
    models.extend_from_slice(o3::ALL);
    models.extend_from_slice(o4::ALL);
    models
}

/// Get all multimodal models (vision + audio capable)
pub fn all_multimodal_models() -> Vec<&'static str> {
    let mut models = Vec::new();
    models.extend_from_slice(gpt_4o::ALL);
    models.extend_from_slice(gpt_4_1::ALL);
    models.extend_from_slice(gpt_4_5::ALL);
    models.extend_from_slice(gpt_5::ALL);
    models
}
