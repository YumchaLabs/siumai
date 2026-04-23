//! Curated Google Vertex model constants aligned with the audited AI SDK package ids
//! plus a small set of provider-owned runtime extras.

/// Google Vertex chat/language-model constants.
pub mod chat {
    pub const GEMINI_2_5_FLASH: &str = "gemini-2.5-flash";
    pub const GEMINI_2_5_PRO: &str = "gemini-2.5-pro";
    pub const GEMINI_2_5_FLASH_LITE: &str = "gemini-2.5-flash-lite";
    pub const GEMINI_2_0_FLASH_LITE: &str = "gemini-2.0-flash-lite";
    pub const GEMINI_2_0_FLASH: &str = "gemini-2.0-flash";
    pub const GEMINI_2_0_FLASH_001: &str = "gemini-2.0-flash-001";
    pub const GEMINI_1_5_FLASH: &str = "gemini-1.5-flash";
    pub const GEMINI_1_5_FLASH_001: &str = "gemini-1.5-flash-001";
    pub const GEMINI_1_5_FLASH_002: &str = "gemini-1.5-flash-002";
    pub const GEMINI_1_5_PRO: &str = "gemini-1.5-pro";
    pub const GEMINI_1_5_PRO_001: &str = "gemini-1.5-pro-001";
    pub const GEMINI_1_5_PRO_002: &str = "gemini-1.5-pro-002";
    pub const GEMINI_1_0_PRO_001: &str = "gemini-1.0-pro-001";
    pub const GEMINI_1_0_PRO_VISION_001: &str = "gemini-1.0-pro-vision-001";
    pub const GEMINI_1_0_PRO: &str = "gemini-1.0-pro";
    pub const GEMINI_1_0_PRO_002: &str = "gemini-1.0-pro-002";
    pub const GEMINI_2_0_FLASH_LITE_PREVIEW_02_05: &str = "gemini-2.0-flash-lite-preview-02-05";
    pub const GEMINI_2_5_FLASH_PREVIEW_09_2025: &str = "gemini-2.5-flash-preview-09-2025";
    pub const GEMINI_3_PRO_PREVIEW: &str = "gemini-3-pro-preview";
    pub const GEMINI_3_PRO_IMAGE_PREVIEW: &str = "gemini-3-pro-image-preview";
    pub const GEMINI_3_FLASH_PREVIEW: &str = "gemini-3-flash-preview";
    pub const GEMINI_3_1_PRO_PREVIEW: &str = "gemini-3.1-pro-preview";
    pub const GEMINI_3_1_FLASH_IMAGE_PREVIEW: &str = "gemini-3.1-flash-image-preview";
    pub const GEMINI_3_1_FLASH_LITE_PREVIEW: &str = "gemini-3.1-flash-lite-preview";
    pub const GEMINI_2_0_PRO_EXP_02_05: &str = "gemini-2.0-pro-exp-02-05";
    pub const GEMINI_2_0_FLASH_EXP: &str = "gemini-2.0-flash-exp";
}

/// Google Vertex embedding-model constants.
pub mod embedding {
    pub const TEXTEMBEDDING_GECKO: &str = "textembedding-gecko";
    pub const TEXTEMBEDDING_GECKO_001: &str = "textembedding-gecko@001";
    pub const TEXTEMBEDDING_GECKO_003: &str = "textembedding-gecko@003";
    pub const TEXTEMBEDDING_GECKO_MULTILINGUAL: &str = "textembedding-gecko-multilingual";
    pub const TEXTEMBEDDING_GECKO_MULTILINGUAL_001: &str = "textembedding-gecko-multilingual@001";
    pub const TEXT_MULTILINGUAL_EMBEDDING_002: &str = "text-multilingual-embedding-002";
    pub const TEXT_EMBEDDING_004: &str = "text-embedding-004";
    pub const TEXT_EMBEDDING_005: &str = "text-embedding-005";
    pub const GEMINI_EMBEDDING_001: &str = "gemini-embedding-001";
    pub const GEMINI_EMBEDDING_2_PREVIEW: &str = "gemini-embedding-2-preview";
}

/// Google Vertex image-model constants.
pub mod image {
    pub const IMAGEN_3_0_GENERATE_001: &str = "imagen-3.0-generate-001";
    pub const IMAGEN_3_0_GENERATE_002: &str = "imagen-3.0-generate-002";
    pub const IMAGEN_3_0_FAST_GENERATE_001: &str = "imagen-3.0-fast-generate-001";
    pub const IMAGEN_4_0_GENERATE_001: &str = "imagen-4.0-generate-001";
    pub const IMAGEN_4_0_ULTRA_GENERATE_001: &str = "imagen-4.0-ultra-generate-001";
    pub const IMAGEN_4_0_FAST_GENERATE_001: &str = "imagen-4.0-fast-generate-001";
    pub const GEMINI_2_5_FLASH_IMAGE: &str = "gemini-2.5-flash-image";
    pub const GEMINI_3_PRO_IMAGE_PREVIEW: &str = "gemini-3-pro-image-preview";
    pub const GEMINI_3_1_FLASH_IMAGE_PREVIEW: &str = "gemini-3.1-flash-image-preview";
    // Provider-owned runtime extra: Siumai still exposes the dedicated Imagen edit model id.
    pub const IMAGEN_3_0_EDIT_001: &str = "imagen-3.0-edit-001";
}

/// Google Vertex video-model constants.
pub mod video {
    pub const VEO_3_1_GENERATE_PREVIEW: &str = "veo-3.1-generate-preview";
    pub const VEO_3_1_FAST_GENERATE_PREVIEW: &str = "veo-3.1-fast-generate-preview";
    pub const VEO_3_1_GENERATE_001: &str = "veo-3.1-generate-001";
    pub const VEO_3_1_FAST_GENERATE_001: &str = "veo-3.1-fast-generate-001";
    pub const VEO_3_0_GENERATE_PREVIEW: &str = "veo-3.0-generate-preview";
    pub const VEO_3_0_FAST_GENERATE_PREVIEW: &str = "veo-3.0-fast-generate-preview";
    pub const VEO_3_0_GENERATE_001: &str = "veo-3.0-generate-001";
    pub const VEO_3_0_FAST_GENERATE_001: &str = "veo-3.0-fast-generate-001";
    pub const VEO_2_0_GENERATE_PREVIEW: &str = "veo-2.0-generate-preview";
    pub const VEO_2_0_GENERATE_EXP: &str = "veo-2.0-generate-exp";
    pub const VEO_2_0_GENERATE_001: &str = "veo-2.0-generate-001";
}

pub const CHAT: &str = chat::GEMINI_2_5_FLASH;
pub const EMBEDDING: &str = embedding::TEXT_EMBEDDING_004;
pub const IMAGE: &str = image::IMAGEN_3_0_GENERATE_002;
pub const VIDEO: &str = video::VEO_3_1_GENERATE_PREVIEW;

pub const ALL_CHAT: &[&str] = &[
    chat::GEMINI_2_5_FLASH,
    chat::GEMINI_2_5_PRO,
    chat::GEMINI_2_5_FLASH_LITE,
    chat::GEMINI_2_0_FLASH_LITE,
    chat::GEMINI_2_0_FLASH,
    chat::GEMINI_2_0_FLASH_001,
    chat::GEMINI_1_5_FLASH,
    chat::GEMINI_1_5_FLASH_001,
    chat::GEMINI_1_5_FLASH_002,
    chat::GEMINI_1_5_PRO,
    chat::GEMINI_1_5_PRO_001,
    chat::GEMINI_1_5_PRO_002,
    chat::GEMINI_1_0_PRO_001,
    chat::GEMINI_1_0_PRO_VISION_001,
    chat::GEMINI_1_0_PRO,
    chat::GEMINI_1_0_PRO_002,
    chat::GEMINI_2_0_FLASH_LITE_PREVIEW_02_05,
    chat::GEMINI_2_5_FLASH_PREVIEW_09_2025,
    chat::GEMINI_3_PRO_PREVIEW,
    chat::GEMINI_3_PRO_IMAGE_PREVIEW,
    chat::GEMINI_3_FLASH_PREVIEW,
    chat::GEMINI_3_1_PRO_PREVIEW,
    chat::GEMINI_3_1_FLASH_IMAGE_PREVIEW,
    chat::GEMINI_3_1_FLASH_LITE_PREVIEW,
    chat::GEMINI_2_0_PRO_EXP_02_05,
    chat::GEMINI_2_0_FLASH_EXP,
];
pub const ALL_EMBEDDING: &[&str] = &[
    embedding::TEXTEMBEDDING_GECKO,
    embedding::TEXTEMBEDDING_GECKO_001,
    embedding::TEXTEMBEDDING_GECKO_003,
    embedding::TEXTEMBEDDING_GECKO_MULTILINGUAL,
    embedding::TEXTEMBEDDING_GECKO_MULTILINGUAL_001,
    embedding::TEXT_MULTILINGUAL_EMBEDDING_002,
    embedding::TEXT_EMBEDDING_004,
    embedding::TEXT_EMBEDDING_005,
    embedding::GEMINI_EMBEDDING_001,
    embedding::GEMINI_EMBEDDING_2_PREVIEW,
];
pub const ALL_IMAGE: &[&str] = &[
    image::IMAGEN_3_0_GENERATE_001,
    image::IMAGEN_3_0_GENERATE_002,
    image::IMAGEN_3_0_FAST_GENERATE_001,
    image::IMAGEN_4_0_GENERATE_001,
    image::IMAGEN_4_0_ULTRA_GENERATE_001,
    image::IMAGEN_4_0_FAST_GENERATE_001,
    image::GEMINI_2_5_FLASH_IMAGE,
    image::GEMINI_3_PRO_IMAGE_PREVIEW,
    image::GEMINI_3_1_FLASH_IMAGE_PREVIEW,
    image::IMAGEN_3_0_EDIT_001,
];
pub const ALL_VIDEO: &[&str] = &[
    video::VEO_3_1_GENERATE_PREVIEW,
    video::VEO_3_1_FAST_GENERATE_PREVIEW,
    video::VEO_3_1_GENERATE_001,
    video::VEO_3_1_FAST_GENERATE_001,
    video::VEO_3_0_GENERATE_PREVIEW,
    video::VEO_3_0_FAST_GENERATE_PREVIEW,
    video::VEO_3_0_GENERATE_001,
    video::VEO_3_0_FAST_GENERATE_001,
    video::VEO_2_0_GENERATE_PREVIEW,
    video::VEO_2_0_GENERATE_EXP,
    video::VEO_2_0_GENERATE_001,
];

fn push_unique(models: &mut Vec<String>, model: &str) {
    if !models.iter().any(|existing| existing == model) {
        models.push(model.to_string());
    }
}

/// Get the default Google Vertex model id set used for public introspection.
pub fn get_default_models() -> Vec<String> {
    let mut models = Vec::new();
    for model in ALL_CHAT
        .iter()
        .chain(ALL_EMBEDDING.iter())
        .chain(ALL_IMAGE.iter())
        .chain(ALL_VIDEO.iter())
    {
        push_unique(&mut models, model);
    }
    models
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curated_defaults_match_current_runtime_defaults() {
        assert_eq!(CHAT, chat::GEMINI_2_5_FLASH);
        assert_eq!(EMBEDDING, embedding::TEXT_EMBEDDING_004);
        assert_eq!(IMAGE, image::IMAGEN_3_0_GENERATE_002);
        assert_eq!(VIDEO, video::VEO_3_1_GENERATE_PREVIEW);
    }

    #[test]
    fn curated_lists_include_primary_defaults() {
        assert!(ALL_CHAT.contains(&CHAT));
        assert!(ALL_EMBEDDING.contains(&EMBEDDING));
        assert!(ALL_IMAGE.contains(&IMAGE));
        assert!(ALL_VIDEO.contains(&VIDEO));
    }

    #[test]
    fn curated_lists_cover_current_audited_vertex_package_ids() {
        assert!(ALL_CHAT.contains(&chat::GEMINI_3_PRO_PREVIEW));
        assert!(ALL_CHAT.contains(&chat::GEMINI_3_1_FLASH_IMAGE_PREVIEW));
        assert!(ALL_EMBEDDING.contains(&embedding::TEXT_EMBEDDING_005));
        assert!(ALL_EMBEDDING.contains(&embedding::GEMINI_EMBEDDING_2_PREVIEW));
        assert!(ALL_IMAGE.contains(&image::IMAGEN_3_0_GENERATE_001));
        assert!(ALL_IMAGE.contains(&image::IMAGEN_4_0_ULTRA_GENERATE_001));
        assert!(ALL_IMAGE.contains(&image::GEMINI_2_5_FLASH_IMAGE));
        assert!(ALL_IMAGE.contains(&image::IMAGEN_3_0_EDIT_001));
        assert!(ALL_VIDEO.contains(&video::VEO_3_1_FAST_GENERATE_001));
    }

    #[test]
    fn default_models_merge_family_sets_without_duplicates() {
        let models = get_default_models();
        assert!(models.iter().any(|model| model == chat::GEMINI_2_5_FLASH));
        assert!(
            models
                .iter()
                .any(|model| model == embedding::TEXT_EMBEDDING_005)
        );
        assert!(
            models
                .iter()
                .any(|model| model == image::GEMINI_3_PRO_IMAGE_PREVIEW)
        );
        assert!(
            models
                .iter()
                .any(|model| model == video::VEO_3_1_GENERATE_PREVIEW)
        );

        let image_preview_count = models
            .iter()
            .filter(|model| model.as_str() == image::GEMINI_3_PRO_IMAGE_PREVIEW)
            .count();
        assert_eq!(image_preview_count, 1);
    }
}
