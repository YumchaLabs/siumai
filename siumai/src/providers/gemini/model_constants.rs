//! Google Gemini Model Constants
//!
//! This module provides convenient constants for Google Gemini models, making it easy
//! for developers to reference specific models without hardcoding strings.

/// Gemini 2.5 Pro model family constants (latest flagship)
pub mod gemini_2_5_pro {
    /// Gemini 2.5 Pro - Most powerful thinking model
    pub const GEMINI_2_5_PRO: &str = "gemini-2.5-pro";

    /// Gemini 2.5 Pro Experimental (03-25)
    /// Aligns with LobeChat provider config id: `gemini-2.5-pro-exp-03-25`
    pub const GEMINI_2_5_PRO_EXP_03_25: &str = "gemini-2.5-pro-exp-03-25";

    /// All Gemini 2.5 Pro models
    pub const ALL: &[&str] = &[GEMINI_2_5_PRO, GEMINI_2_5_PRO_EXP_03_25];
}

/// Gemini 2.5 Flash model family constants
pub mod gemini_2_5_flash {
    /// Gemini 2.5 Flash - Excellent cost-performance ratio
    pub const GEMINI_2_5_FLASH: &str = "gemini-2.5-flash";
    /// Gemini 2.5 Flash Preview (05-20)
    pub const GEMINI_2_5_FLASH_PREVIEW_05_20: &str = "gemini-2.5-flash-preview-05-20";

    /// All Gemini 2.5 Flash models
    pub const ALL: &[&str] = &[GEMINI_2_5_FLASH, GEMINI_2_5_FLASH_PREVIEW_05_20];
}

/// Gemini 2.5 Flash-Lite model family constants
pub mod gemini_2_5_flash_lite {
    /// Gemini 2.5 Flash-Lite - Optimized for cost-effectiveness and high throughput
    pub const GEMINI_2_5_FLASH_LITE: &str = "gemini-2.5-flash-lite";
    /// Gemini 2.5 Flash-Lite Preview (06-17)
    pub const GEMINI_2_5_FLASH_LITE_06_17: &str = "gemini-2.5-flash-lite-06-17";

    /// All Gemini 2.5 Flash-Lite models
    pub const ALL: &[&str] = &[GEMINI_2_5_FLASH_LITE, GEMINI_2_5_FLASH_LITE_06_17];
}

/// Gemini 2.5 Flash Live model family constants
pub mod gemini_2_5_flash_live {
    /// Gemini 2.5 Flash Live - Low-latency bidirectional voice and video interaction
    pub const GEMINI_LIVE_2_5_FLASH_PREVIEW: &str = "gemini-live-2.5-flash-preview";

    /// All Gemini 2.5 Flash Live models
    pub const ALL: &[&str] = &[GEMINI_LIVE_2_5_FLASH_PREVIEW];
}

/// Gemini 2.5 Flash Native Audio model family constants
pub mod gemini_2_5_flash_native_audio {
    /// Gemini 2.5 Flash Native Audio Dialog
    pub const GEMINI_2_5_FLASH_PREVIEW_NATIVE_AUDIO_DIALOG: &str =
        "gemini-2.5-flash-preview-native-audio-dialog";
    /// Gemini 2.5 Flash Native Audio Thinking Dialog (experimental)
    pub const GEMINI_2_5_FLASH_EXP_NATIVE_AUDIO_THINKING_DIALOG: &str =
        "gemini-2.5-flash-exp-native-audio-thinking-dialog";

    /// All Gemini 2.5 Flash Native Audio models
    pub const ALL: &[&str] = &[
        GEMINI_2_5_FLASH_PREVIEW_NATIVE_AUDIO_DIALOG,
        GEMINI_2_5_FLASH_EXP_NATIVE_AUDIO_THINKING_DIALOG,
    ];
}

/// Gemini 2.5 TTS model family constants
pub mod gemini_2_5_tts {
    /// Gemini 2.5 Flash Preview TTS
    pub const GEMINI_2_5_FLASH_PREVIEW_TTS: &str = "gemini-2.5-flash-preview-tts";
    /// Gemini 2.5 Pro Preview TTS
    pub const GEMINI_2_5_PRO_PREVIEW_TTS: &str = "gemini-2.5-pro-preview-tts";

    /// All Gemini 2.5 TTS models
    pub const ALL: &[&str] = &[GEMINI_2_5_FLASH_PREVIEW_TTS, GEMINI_2_5_PRO_PREVIEW_TTS];
}

/// Gemini 2.0 Flash model family constants
pub mod gemini_2_0_flash {
    /// Gemini 2.0 Flash - Next-generation features and improved performance
    pub const GEMINI_2_0_FLASH: &str = "gemini-2.0-flash";
    /// Gemini 2.0 Flash (001)
    pub const GEMINI_2_0_FLASH_001: &str = "gemini-2.0-flash-001";
    /// Gemini 2.0 Flash Experimental
    pub const GEMINI_2_0_FLASH_EXP: &str = "gemini-2.0-flash-exp";

    /// All Gemini 2.0 Flash models
    pub const ALL: &[&str] = &[GEMINI_2_0_FLASH, GEMINI_2_0_FLASH_001, GEMINI_2_0_FLASH_EXP];
}

/// Gemini 2.0 Pro model family constants
pub mod gemini_2_0_pro {
    /// Gemini 2.0 Pro Experimental (02-05)
    /// Aligns with LobeChat provider config id: `gemini-2.0-pro-exp-02-05`
    pub const GEMINI_2_0_PRO_EXP_02_05: &str = "gemini-2.0-pro-exp-02-05";

    /// All Gemini 2.0 Pro models
    pub const ALL: &[&str] = &[GEMINI_2_0_PRO_EXP_02_05];
}

/// Gemini 2.0 Flash Image Generation model family constants
pub mod gemini_2_0_flash_image_gen {
    /// Gemini 2.0 Flash Preview Image Generation
    pub const GEMINI_2_0_FLASH_PREVIEW_IMAGE_GENERATION: &str =
        "gemini-2.0-flash-preview-image-generation";

    /// All Gemini 2.0 Flash Image Generation models
    pub const ALL: &[&str] = &[GEMINI_2_0_FLASH_PREVIEW_IMAGE_GENERATION];
}

/// Gemini 2.0 Flash-Lite model family constants
pub mod gemini_2_0_flash_lite {
    /// Gemini 2.0 Flash-Lite - Optimized for cost-effectiveness and reduced latency
    pub const GEMINI_2_0_FLASH_LITE: &str = "gemini-2.0-flash-lite";
    /// Gemini 2.0 Flash-Lite (001)
    pub const GEMINI_2_0_FLASH_LITE_001: &str = "gemini-2.0-flash-lite-001";
    /// Gemini 2.0 Flash-Lite Preview (02-05)
    pub const GEMINI_2_0_FLASH_LITE_PREVIEW_02_05: &str = "gemini-2.0-flash-lite-preview-02-05";

    /// All Gemini 2.0 Flash-Lite models
    pub const ALL: &[&str] = &[
        GEMINI_2_0_FLASH_LITE,
        GEMINI_2_0_FLASH_LITE_001,
        GEMINI_2_0_FLASH_LITE_PREVIEW_02_05,
    ];
}

/// Gemini 2.0 Flash Thinking (experimental) model family constants
pub mod gemini_2_0_flash_thinking_exp {
    /// Gemini 2.0 Flash Thinking Experimental (01-21)
    /// Aligns with LobeChat provider config id: `gemini-2.0-flash-thinking-exp-01-21`
    pub const GEMINI_2_0_FLASH_THINKING_EXP_01_21: &str = "gemini-2.0-flash-thinking-exp-01-21";

    /// All Gemini 2.0 Flash Thinking Experimental models
    pub const ALL: &[&str] = &[GEMINI_2_0_FLASH_THINKING_EXP_01_21];
}

/// Gemini 2.0 Flash Live model family constants
pub mod gemini_2_0_flash_live {
    /// Gemini 2.0 Flash Live - Low-latency bidirectional voice and video interaction
    pub const GEMINI_2_0_FLASH_LIVE_001: &str = "gemini-2.0-flash-live-001";

    /// All Gemini 2.0 Flash Live models
    pub const ALL: &[&str] = &[GEMINI_2_0_FLASH_LIVE_001];
}

/// Gemini 1.5 Flash model family constants (deprecated)
pub mod gemini_1_5_flash {
    /// Gemini 1.5 Flash - Fast and versatile multimodal model
    pub const GEMINI_1_5_FLASH: &str = "gemini-1.5-flash";
    /// Gemini 1.5 Flash Latest
    pub const GEMINI_1_5_FLASH_LATEST: &str = "gemini-1.5-flash-latest";
    /// Gemini 1.5 Flash (001)
    pub const GEMINI_1_5_FLASH_001: &str = "gemini-1.5-flash-001";
    /// Gemini 1.5 Flash (002)
    pub const GEMINI_1_5_FLASH_002: &str = "gemini-1.5-flash-002";

    /// All Gemini 1.5 Flash models
    pub const ALL: &[&str] = &[
        GEMINI_1_5_FLASH,
        GEMINI_1_5_FLASH_LATEST,
        GEMINI_1_5_FLASH_001,
        GEMINI_1_5_FLASH_002,
    ];
}

/// Gemini 1.5 Flash-8B model family constants (deprecated)
pub mod gemini_1_5_flash_8b {
    /// Gemini 1.5 Flash-8B - Small model for high-volume, lower-intelligence tasks
    pub const GEMINI_1_5_FLASH_8B: &str = "gemini-1.5-flash-8b";
    /// Gemini 1.5 Flash-8B Latest
    pub const GEMINI_1_5_FLASH_8B_LATEST: &str = "gemini-1.5-flash-8b-latest";
    /// Gemini 1.5 Flash-8B (001)
    pub const GEMINI_1_5_FLASH_8B_001: &str = "gemini-1.5-flash-8b-001";

    /// All Gemini 1.5 Flash-8B models
    pub const ALL: &[&str] = &[
        GEMINI_1_5_FLASH_8B,
        GEMINI_1_5_FLASH_8B_LATEST,
        GEMINI_1_5_FLASH_8B_001,
    ];
}

/// Gemini 1.5 Pro model family constants (deprecated)
pub mod gemini_1_5_pro {
    /// Gemini 1.5 Pro - Mid-size multimodal model optimized for reasoning tasks
    pub const GEMINI_1_5_PRO: &str = "gemini-1.5-pro";
    /// Gemini 1.5 Pro Latest
    pub const GEMINI_1_5_PRO_LATEST: &str = "gemini-1.5-pro-latest";
    /// Gemini 1.5 Pro (001)
    pub const GEMINI_1_5_PRO_001: &str = "gemini-1.5-pro-001";
    /// Gemini 1.5 Pro (002)
    pub const GEMINI_1_5_PRO_002: &str = "gemini-1.5-pro-002";

    /// All Gemini 1.5 Pro models
    pub const ALL: &[&str] = &[
        GEMINI_1_5_PRO,
        GEMINI_1_5_PRO_LATEST,
        GEMINI_1_5_PRO_001,
        GEMINI_1_5_PRO_002,
    ];
}

/// LearnLM model family constants (experimental)
pub mod learnlm_1_5_pro {
    /// LearnLM 1.5 Pro Experimental
    /// Aligns with LobeChat provider config id: `learnlm-1.5-pro-experimental`
    pub const LEARNLM_1_5_PRO_EXPERIMENTAL: &str = "learnlm-1.5-pro-experimental";

    /// All LearnLM 1.5 Pro models
    pub const ALL: &[&str] = &[LEARNLM_1_5_PRO_EXPERIMENTAL];
}

/// Popular model recommendations
pub mod popular {
    use super::*;

    /// Most capable model
    pub const FLAGSHIP: &str = gemini_2_5_pro::GEMINI_2_5_PRO;
    /// Best balance of capability and cost
    pub const BALANCED: &str = gemini_2_5_flash::GEMINI_2_5_FLASH;
    /// Most cost-effective
    pub const ECONOMICAL: &str = gemini_2_5_flash_lite::GEMINI_2_5_FLASH_LITE;
    /// Best for real-time interaction
    pub const REALTIME: &str = gemini_2_5_flash_live::GEMINI_LIVE_2_5_FLASH_PREVIEW;
    /// Latest and most advanced
    pub const LATEST: &str = gemini_2_5_pro::GEMINI_2_5_PRO;
}

/// Model capabilities by family
pub mod capabilities {
    /// Models with thinking capability
    pub const THINKING_MODELS: &[&str] = &[
        super::gemini_2_5_pro::GEMINI_2_5_PRO,
        super::gemini_2_5_flash::GEMINI_2_5_FLASH,
        super::gemini_2_5_flash_lite::GEMINI_2_5_FLASH_LITE,
    ];

    /// Models with image generation capability
    pub const IMAGE_GENERATION_MODELS: &[&str] =
        &[super::gemini_2_0_flash_image_gen::GEMINI_2_0_FLASH_PREVIEW_IMAGE_GENERATION];

    /// Models with audio generation capability
    pub const AUDIO_GENERATION_MODELS: &[&str] = &[
        super::gemini_2_5_tts::GEMINI_2_5_FLASH_PREVIEW_TTS,
        super::gemini_2_5_tts::GEMINI_2_5_PRO_PREVIEW_TTS,
        super::gemini_2_5_flash_live::GEMINI_LIVE_2_5_FLASH_PREVIEW,
        super::gemini_2_0_flash_live::GEMINI_2_0_FLASH_LIVE_001,
    ];

    /// Models with Live API support
    pub const LIVE_API_MODELS: &[&str] = &[
        super::gemini_2_5_flash_live::GEMINI_LIVE_2_5_FLASH_PREVIEW,
        super::gemini_2_0_flash_live::GEMINI_2_0_FLASH_LIVE_001,
    ];
}

/// Get all chat models
pub fn all_chat_models() -> Vec<&'static str> {
    let mut models = Vec::new();
    models.extend_from_slice(gemini_2_5_pro::ALL);
    models.extend_from_slice(gemini_2_5_flash::ALL);
    models.extend_from_slice(gemini_2_5_flash_lite::ALL);
    models.extend_from_slice(gemini_2_0_flash::ALL);
    models.extend_from_slice(gemini_2_0_pro::ALL);
    models.extend_from_slice(gemini_2_0_flash_thinking_exp::ALL);
    models.extend_from_slice(gemini_2_0_flash_lite::ALL);
    models.extend_from_slice(gemini_1_5_flash::ALL);
    models.extend_from_slice(gemini_1_5_flash_8b::ALL);
    models.extend_from_slice(gemini_1_5_pro::ALL);
    models.extend_from_slice(learnlm_1_5_pro::ALL);
    models
}

/// Get all models with thinking capability
pub fn all_thinking_models() -> Vec<&'static str> {
    capabilities::THINKING_MODELS.to_vec()
}

/// Get all models with image generation capability
pub fn all_image_generation_models() -> Vec<&'static str> {
    capabilities::IMAGE_GENERATION_MODELS.to_vec()
}

/// Get all models with audio generation capability
pub fn all_audio_generation_models() -> Vec<&'static str> {
    capabilities::AUDIO_GENERATION_MODELS.to_vec()
}

/// Get all models with Live API support
pub fn all_live_api_models() -> Vec<&'static str> {
    capabilities::LIVE_API_MODELS.to_vec()
}
