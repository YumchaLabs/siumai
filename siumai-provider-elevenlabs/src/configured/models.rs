//! Evidence-backed ElevenLabs Text-to-Speech model identifiers.

pub const ELEVEN_V3: &str = "eleven_v3";
pub const ELEVEN_MULTILINGUAL_V2: &str = "eleven_multilingual_v2";
pub const ELEVEN_FLASH_V2_5: &str = "eleven_flash_v2_5";
pub const ELEVEN_FLASH_V2: &str = "eleven_flash_v2";
pub const ELEVEN_TURBO_V2_5: &str = "eleven_turbo_v2_5";
pub const ELEVEN_TURBO_V2: &str = "eleven_turbo_v2";
pub const ELEVEN_MULTILINGUAL_V1: &str = "eleven_multilingual_v1";

pub const DEFAULT: &str = ELEVEN_MULTILINGUAL_V2;
pub const DEFAULT_VOICE: &str = "21m00Tcm4TlvDq8ikWAM";

pub const VERIFIED: &[&str] = &[
    ELEVEN_V3,
    ELEVEN_MULTILINGUAL_V2,
    ELEVEN_FLASH_V2_5,
    ELEVEN_FLASH_V2,
    ELEVEN_TURBO_V2_5,
    ELEVEN_TURBO_V2,
    ELEVEN_MULTILINGUAL_V1,
];

/// Provider-published character budgets for one text-to-speech request.
pub(crate) fn max_text_chars(model: &str) -> Option<usize> {
    match model {
        ELEVEN_V3 => Some(5_000),
        ELEVEN_MULTILINGUAL_V2 | ELEVEN_MULTILINGUAL_V1 => Some(10_000),
        ELEVEN_FLASH_V2_5 | ELEVEN_TURBO_V2_5 => Some(40_000),
        ELEVEN_FLASH_V2 | ELEVEN_TURBO_V2 => Some(30_000),
        _ => None,
    }
}
