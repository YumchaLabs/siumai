//! Curated Google Vertex model constants aligned with the audited public subset.

/// Google Vertex chat/language-model constants.
pub mod chat {
    pub const GEMINI_2_5_FLASH: &str = "gemini-2.5-flash";
    pub const GEMINI_2_5_PRO: &str = "gemini-2.5-pro";
    pub const GEMINI_2_0_FLASH: &str = "gemini-2.0-flash";
}

/// Google Vertex embedding-model constants.
pub mod embedding {
    pub const TEXT_EMBEDDING_004: &str = "text-embedding-004";
}

/// Google Vertex image-model constants.
pub mod image {
    pub const IMAGEN_3_0_GENERATE_002: &str = "imagen-3.0-generate-002";
    pub const IMAGEN_4_0_GENERATE_001: &str = "imagen-4.0-generate-001";
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
    chat::GEMINI_2_0_FLASH,
];
pub const ALL_EMBEDDING: &[&str] = &[embedding::TEXT_EMBEDDING_004];
pub const ALL_IMAGE: &[&str] = &[
    image::IMAGEN_3_0_GENERATE_002,
    image::IMAGEN_4_0_GENERATE_001,
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
}
