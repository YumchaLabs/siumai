//! Dated ergonomic hints for current Gemini image models.

/// Primary Nano Banana 2 model.
pub const GEMINI_3_1_FLASH_IMAGE: &str = "gemini-3.1-flash-image";
/// Cost-efficient Nano Banana 2 model.
pub const GEMINI_3_1_FLASH_LITE_IMAGE: &str = "gemini-3.1-flash-lite-image";
/// High-fidelity Nano Banana Pro model.
pub const GEMINI_3_PRO_IMAGE: &str = "gemini-3-pro-image";

/// Current model hints verified from Google documentation on 2026-08-06.
///
/// The list is advisory. Future model IDs remain valid input.
pub const fn current_models() -> [&'static str; 3] {
    [
        GEMINI_3_1_FLASH_IMAGE,
        GEMINI_3_1_FLASH_LITE_IMAGE,
        GEMINI_3_PRO_IMAGE,
    ]
}

pub(crate) fn is_current(model: &str) -> bool {
    current_models().contains(&model)
}
