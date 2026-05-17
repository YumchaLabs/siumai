//! xAI model constants aligned with the audited AI SDK package subset.
/// Grok 4 model family constants.
pub mod grok_4 {
    pub const GROK_4_1_FAST_REASONING: &str = "grok-4-1-fast-reasoning";
    pub const GROK_4_1_FAST_NON_REASONING: &str = "grok-4-1-fast-non-reasoning";
    pub const GROK_4_FAST_NON_REASONING: &str = "grok-4-fast-non-reasoning";
    pub const GROK_4_FAST_REASONING: &str = "grok-4-fast-reasoning";
    pub const GROK_4_20_0309_NON_REASONING: &str = "grok-4.20-0309-non-reasoning";
    pub const GROK_4_20_0309_REASONING: &str = "grok-4.20-0309-reasoning";
    pub const GROK_4_20_MULTI_AGENT_0309: &str = "grok-4.20-multi-agent-0309";
    pub const GROK_4: &str = "grok-4";
    pub const GROK_4_0709: &str = "grok-4-0709";
    pub const GROK_4_LATEST: &str = "grok-4-latest";

    pub const ALL: &[&str] = &[
        GROK_4_1_FAST_REASONING,
        GROK_4_1_FAST_NON_REASONING,
        GROK_4_FAST_NON_REASONING,
        GROK_4_FAST_REASONING,
        GROK_4_20_0309_NON_REASONING,
        GROK_4_20_0309_REASONING,
        GROK_4_20_MULTI_AGENT_0309,
        GROK_4,
        GROK_4_0709,
        GROK_4_LATEST,
    ];
}

/// Grok 3 model family constants.
pub mod grok_3 {
    pub const GROK_3: &str = "grok-3";
    pub const GROK_3_LATEST: &str = "grok-3-latest";
    pub const GROK_3_MINI: &str = "grok-3-mini";
    pub const GROK_3_MINI_LATEST: &str = "grok-3-mini-latest";

    pub const ALL: &[&str] = &[GROK_3, GROK_3_LATEST, GROK_3_MINI, GROK_3_MINI_LATEST];
}

/// Code-specialized Grok model constants.
pub mod code {
    pub const GROK_CODE_FAST_1: &str = "grok-code-fast-1";

    pub const ALL: &[&str] = &[GROK_CODE_FAST_1];
}

/// Native image generation model constants.
pub mod image {
    pub const GROK_IMAGINE_IMAGE: &str = "grok-imagine-image";
    pub const GROK_IMAGINE_IMAGE_PRO: &str = "grok-imagine-image-pro";

    pub const ALL: &[&str] = &[GROK_IMAGINE_IMAGE, GROK_IMAGINE_IMAGE_PRO];
}

/// Native video generation model constants.
pub mod video {
    pub const GROK_IMAGINE_VIDEO: &str = "grok-imagine-video";

    pub const ALL: &[&str] = &[GROK_IMAGINE_VIDEO];
}

/// Legacy xAI model constants preserved for older imports.
pub mod legacy {
    pub const GROK_BETA: &str = "grok-beta";
    pub const GROK_VISION_BETA: &str = "grok-vision-beta";

    pub const ALL: &[&str] = &[GROK_BETA, GROK_VISION_BETA];
}

pub const CHAT: &str = grok_4::GROK_4;
pub const IMAGE: &str = image::GROK_IMAGINE_IMAGE;
pub const VIDEO: &str = video::GROK_IMAGINE_VIDEO;

pub const ALL_CHAT: &[&str] = &[
    grok_4::GROK_4_1_FAST_REASONING,
    grok_4::GROK_4_1_FAST_NON_REASONING,
    grok_4::GROK_4_FAST_NON_REASONING,
    grok_4::GROK_4_FAST_REASONING,
    grok_4::GROK_4_20_0309_NON_REASONING,
    grok_4::GROK_4_20_0309_REASONING,
    grok_4::GROK_4_20_MULTI_AGENT_0309,
    grok_4::GROK_4,
    grok_4::GROK_4_0709,
    grok_4::GROK_4_LATEST,
    grok_3::GROK_3,
    grok_3::GROK_3_LATEST,
    grok_3::GROK_3_MINI,
    grok_3::GROK_3_MINI_LATEST,
    code::GROK_CODE_FAST_1,
];

pub const ALL_IMAGE: &[&str] = image::ALL;
pub const ALL_VIDEO: &[&str] = video::ALL;

pub mod popular {
    use super::*;

    pub const FLAGSHIP: &str = grok_4::GROK_4;
    pub const FAST: &str = grok_4::GROK_4_1_FAST_NON_REASONING;
    pub const REASONING: &str = grok_4::GROK_4_1_FAST_REASONING;
    pub const CODING: &str = code::GROK_CODE_FAST_1;
    pub const IMAGE_GENERATION: &str = image::GROK_IMAGINE_IMAGE;
    pub const VIDEO_GENERATION: &str = video::GROK_IMAGINE_VIDEO;
}

pub use code::GROK_CODE_FAST_1;
pub use grok_3::GROK_3;
pub use grok_3::GROK_3_MINI;
pub use grok_4::GROK_4;
pub use image::GROK_IMAGINE_IMAGE;
pub use legacy::GROK_BETA;
pub use legacy::GROK_VISION_BETA;

pub fn all_models() -> Vec<String> {
    let mut models = Vec::new();
    models.extend(ALL_CHAT.iter().map(|&model| model.to_string()));
    models.extend(ALL_IMAGE.iter().map(|&model| model.to_string()));
    models.extend(ALL_VIDEO.iter().map(|&model| model.to_string()));
    models.extend(legacy::ALL.iter().map(|&model| model.to_string()));
    models
}
