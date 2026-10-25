//! Dated Gemini model hints, not an allowlist or an API-version support matrix.

/// Model hint verified on 2026-10-25; not yet a stable-v1 catalog claim.
pub const GEMINI_3_8_FLASH: &str = "gemini-3.8-flash";
/// Model hint verified on 2026-10-25; not yet a stable-v1 catalog claim.
pub const GEMINI_4_ARGON: &str = "gemini-4-argon";
/// Image-model hint verified on 2026-10-25; not yet a stable-v1 catalog claim.
pub const GEMINI_NANO_BANANA_2_1: &str = "gemini-nano-banana-2.1";

/// Current stable Interactions language model shown in Google's v1 reference.
pub const GEMINI_3_6_FLASH: &str = "gemini-3.6-flash";
/// Gemini 3.5 Flash, deprecated in favor of Gemini 3.6 Flash on 2026-10-08.
pub const GEMINI_3_5_FLASH: &str = "gemini-3.5-flash";
/// Current cost-efficient Gemini 3.5 Flash Lite model hint.
pub const GEMINI_3_5_FLASH_LITE: &str = "gemini-3.5-flash-lite";

/// Nano Banana 2, deprecated in favor of Nano Banana 2.1 on 2026-10-06.
pub const GEMINI_3_1_FLASH_IMAGE: &str = "gemini-3.1-flash-image";
/// Cost-efficient Nano Banana 2 model.
pub const GEMINI_3_1_FLASH_LITE_IMAGE: &str = "gemini-3.1-flash-lite-image";
/// High-fidelity Nano Banana Pro model.
pub const GEMINI_3_PRO_IMAGE: &str = "gemini-3-pro-image";

/// Current image-model hints verified from Google documentation on 2026-08-08.
///
/// The list is advisory. Future model IDs remain valid input.
pub const fn current_image_models() -> [&'static str; 3] {
    [
        GEMINI_3_1_FLASH_IMAGE,
        GEMINI_3_1_FLASH_LITE_IMAGE,
        GEMINI_3_PRO_IMAGE,
    ]
}

/// Current Interactions language-model hints verified from Google documentation on 2026-08-08.
///
/// The list is advisory. Future model IDs remain valid input.
pub const fn current_interactions_models() -> [&'static str; 3] {
    [GEMINI_3_6_FLASH, GEMINI_3_5_FLASH, GEMINI_3_5_FLASH_LITE]
}
