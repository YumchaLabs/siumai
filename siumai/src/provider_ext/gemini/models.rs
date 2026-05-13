/// Google/Gemini language-model ids exported by the audited AI SDK package.
pub mod chat {
    pub const GEMINI_2_0_FLASH: &str = "gemini-2.0-flash";
    pub const GEMINI_2_0_FLASH_001: &str = "gemini-2.0-flash-001";
    pub const GEMINI_2_0_FLASH_LITE: &str = "gemini-2.0-flash-lite";
    pub const GEMINI_2_0_FLASH_LITE_001: &str = "gemini-2.0-flash-lite-001";
    pub const GEMINI_2_5_PRO: &str = "gemini-2.5-pro";
    pub const GEMINI_2_5_FLASH: &str = "gemini-2.5-flash";
    pub const GEMINI_2_5_FLASH_IMAGE: &str = "gemini-2.5-flash-image";
    pub const GEMINI_2_5_FLASH_LITE: &str = "gemini-2.5-flash-lite";
    pub const GEMINI_2_5_FLASH_PREVIEW_TTS: &str = "gemini-2.5-flash-preview-tts";
    pub const GEMINI_2_5_PRO_PREVIEW_TTS: &str = "gemini-2.5-pro-preview-tts";
    pub const GEMINI_2_5_FLASH_NATIVE_AUDIO_LATEST: &str = "gemini-2.5-flash-native-audio-latest";
    pub const GEMINI_2_5_FLASH_NATIVE_AUDIO_PREVIEW_09_2025: &str =
        "gemini-2.5-flash-native-audio-preview-09-2025";
    pub const GEMINI_2_5_FLASH_NATIVE_AUDIO_PREVIEW_12_2025: &str =
        "gemini-2.5-flash-native-audio-preview-12-2025";
    pub const GEMINI_2_5_COMPUTER_USE_PREVIEW_10_2025: &str =
        "gemini-2.5-computer-use-preview-10-2025";
    pub const GEMINI_3_PRO_PREVIEW: &str = "gemini-3-pro-preview";
    pub const GEMINI_3_PRO_IMAGE_PREVIEW: &str = "gemini-3-pro-image-preview";
    pub const GEMINI_3_FLASH_PREVIEW: &str = "gemini-3-flash-preview";
    pub const GEMINI_3_1_PRO_PREVIEW: &str = "gemini-3.1-pro-preview";
    pub const GEMINI_3_1_PRO_PREVIEW_CUSTOMTOOLS: &str = "gemini-3.1-pro-preview-customtools";
    pub const GEMINI_3_1_FLASH_IMAGE_PREVIEW: &str = "gemini-3.1-flash-image-preview";
    pub const GEMINI_3_1_FLASH_LITE_PREVIEW: &str = "gemini-3.1-flash-lite-preview";
    pub const GEMINI_3_1_FLASH_TTS_PREVIEW: &str = "gemini-3.1-flash-tts-preview";
    pub const GEMINI_PRO_LATEST: &str = "gemini-pro-latest";
    pub const GEMINI_FLASH_LATEST: &str = "gemini-flash-latest";
    pub const GEMINI_FLASH_LITE_LATEST: &str = "gemini-flash-lite-latest";
    pub const DEEP_RESEARCH_PRO_PREVIEW_12_2025: &str = "deep-research-pro-preview-12-2025";
    pub const NANO_BANANA_PRO_PREVIEW: &str = "nano-banana-pro-preview";
    pub const AQA: &str = "aqa";
    pub const GEMINI_ROBOTICS_ER_1_5_PREVIEW: &str = "gemini-robotics-er-1.5-preview";
    pub const GEMMA_3_1B_IT: &str = "gemma-3-1b-it";
    pub const GEMMA_3_4B_IT: &str = "gemma-3-4b-it";
    pub const GEMMA_3N_E4B_IT: &str = "gemma-3n-e4b-it";
    pub const GEMMA_3N_E2B_IT: &str = "gemma-3n-e2b-it";
    pub const GEMMA_3_12B_IT: &str = "gemma-3-12b-it";
    pub const GEMMA_3_27B_IT: &str = "gemma-3-27b-it";

    pub const ALL: &[&str] = &[
        GEMINI_2_0_FLASH,
        GEMINI_2_0_FLASH_001,
        GEMINI_2_0_FLASH_LITE,
        GEMINI_2_0_FLASH_LITE_001,
        GEMINI_2_5_PRO,
        GEMINI_2_5_FLASH,
        GEMINI_2_5_FLASH_IMAGE,
        GEMINI_2_5_FLASH_LITE,
        GEMINI_2_5_FLASH_PREVIEW_TTS,
        GEMINI_2_5_PRO_PREVIEW_TTS,
        GEMINI_2_5_FLASH_NATIVE_AUDIO_LATEST,
        GEMINI_2_5_FLASH_NATIVE_AUDIO_PREVIEW_09_2025,
        GEMINI_2_5_FLASH_NATIVE_AUDIO_PREVIEW_12_2025,
        GEMINI_2_5_COMPUTER_USE_PREVIEW_10_2025,
        GEMINI_3_PRO_PREVIEW,
        GEMINI_3_PRO_IMAGE_PREVIEW,
        GEMINI_3_FLASH_PREVIEW,
        GEMINI_3_1_PRO_PREVIEW,
        GEMINI_3_1_PRO_PREVIEW_CUSTOMTOOLS,
        GEMINI_3_1_FLASH_IMAGE_PREVIEW,
        GEMINI_3_1_FLASH_LITE_PREVIEW,
        GEMINI_3_1_FLASH_TTS_PREVIEW,
        GEMINI_PRO_LATEST,
        GEMINI_FLASH_LATEST,
        GEMINI_FLASH_LITE_LATEST,
        DEEP_RESEARCH_PRO_PREVIEW_12_2025,
        NANO_BANANA_PRO_PREVIEW,
        AQA,
        GEMINI_ROBOTICS_ER_1_5_PREVIEW,
        GEMMA_3_1B_IT,
        GEMMA_3_4B_IT,
        GEMMA_3N_E4B_IT,
        GEMMA_3N_E2B_IT,
        GEMMA_3_12B_IT,
        GEMMA_3_27B_IT,
    ];
}

/// Google embedding-model ids exported by the audited AI SDK package.
pub mod embedding {
    pub const GEMINI_EMBEDDING_001: &str = "gemini-embedding-001";
    pub const GEMINI_EMBEDDING_2_PREVIEW: &str = "gemini-embedding-2-preview";

    pub const ALL: &[&str] = &[GEMINI_EMBEDDING_001, GEMINI_EMBEDDING_2_PREVIEW];
}

/// Google image-model ids exported by the audited AI SDK package.
pub mod image {
    pub const IMAGEN_4_0_GENERATE_001: &str = "imagen-4.0-generate-001";
    pub const IMAGEN_4_0_ULTRA_GENERATE_001: &str = "imagen-4.0-ultra-generate-001";
    pub const IMAGEN_4_0_FAST_GENERATE_001: &str = "imagen-4.0-fast-generate-001";
    pub const GEMINI_2_5_FLASH_IMAGE: &str = "gemini-2.5-flash-image";
    pub const GEMINI_3_PRO_IMAGE_PREVIEW: &str = "gemini-3-pro-image-preview";
    pub const GEMINI_3_1_FLASH_IMAGE_PREVIEW: &str = "gemini-3.1-flash-image-preview";

    pub const ALL: &[&str] = &[
        IMAGEN_4_0_GENERATE_001,
        IMAGEN_4_0_ULTRA_GENERATE_001,
        IMAGEN_4_0_FAST_GENERATE_001,
        GEMINI_2_5_FLASH_IMAGE,
        GEMINI_3_PRO_IMAGE_PREVIEW,
        GEMINI_3_1_FLASH_IMAGE_PREVIEW,
    ];
}

/// Google video-model ids exported by the audited AI SDK package.
pub mod video {
    pub const VEO_3_1_FAST_GENERATE_PREVIEW: &str = "veo-3.1-fast-generate-preview";
    pub const VEO_3_1_GENERATE_PREVIEW: &str = "veo-3.1-generate-preview";
    pub const VEO_3_1_GENERATE: &str = "veo-3.1-generate";
    pub const VEO_3_1_LITE_GENERATE_PREVIEW: &str = "veo-3.1-lite-generate-preview";
    pub const VEO_3_0_GENERATE_001: &str = "veo-3.0-generate-001";
    pub const VEO_3_0_FAST_GENERATE_001: &str = "veo-3.0-fast-generate-001";
    pub const VEO_2_0_GENERATE_001: &str = "veo-2.0-generate-001";

    pub const ALL: &[&str] = &[
        VEO_3_1_FAST_GENERATE_PREVIEW,
        VEO_3_1_GENERATE_PREVIEW,
        VEO_3_1_GENERATE,
        VEO_3_1_LITE_GENERATE_PREVIEW,
        VEO_3_0_GENERATE_001,
        VEO_3_0_FAST_GENERATE_001,
        VEO_2_0_GENERATE_001,
    ];
}

/// Group aliases that mirror the common public provider-facade convention.
pub mod model_sets {
    pub const ALL_CHAT: &[&str] = super::chat::ALL;
    pub const ALL_EMBEDDING: &[&str] = super::embedding::ALL;
    pub const ALL_IMAGE: &[&str] = super::image::ALL;
    pub const ALL_VIDEO: &[&str] = super::video::ALL;

    pub const CHAT: &[&str] = ALL_CHAT;
    pub const EMBEDDING: &[&str] = ALL_EMBEDDING;
    pub const IMAGE: &[&str] = ALL_IMAGE;
    pub const VIDEO: &[&str] = ALL_VIDEO;
}
