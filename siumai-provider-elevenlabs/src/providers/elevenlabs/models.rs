//! ElevenLabs model catalog aligned with `@ai-sdk/elevenlabs`.

/// ElevenLabs speech model ids.
pub mod speech {
    pub const ELEVEN_V3: &str = "eleven_v3";
    pub const ELEVEN_MULTILINGUAL_V2: &str = "eleven_multilingual_v2";
    pub const ELEVEN_FLASH_V2_5: &str = "eleven_flash_v2_5";
    pub const ELEVEN_FLASH_V2: &str = "eleven_flash_v2";
    pub const ELEVEN_TURBO_V2_5: &str = "eleven_turbo_v2_5";
    pub const ELEVEN_TURBO_V2: &str = "eleven_turbo_v2";
    pub const ELEVEN_MONOLINGUAL_V1: &str = "eleven_monolingual_v1";
    pub const ELEVEN_MULTILINGUAL_V1: &str = "eleven_multilingual_v1";

    pub const ALL: &[&str] = &[
        ELEVEN_V3,
        ELEVEN_MULTILINGUAL_V2,
        ELEVEN_FLASH_V2_5,
        ELEVEN_FLASH_V2,
        ELEVEN_TURBO_V2_5,
        ELEVEN_TURBO_V2,
        ELEVEN_MONOLINGUAL_V1,
        ELEVEN_MULTILINGUAL_V1,
    ];
}

/// ElevenLabs transcription model ids.
pub mod transcription {
    pub const SCRIBE_V1: &str = "scribe_v1";
    pub const SCRIBE_V1_EXPERIMENTAL: &str = "scribe_v1_experimental";

    pub const ALL: &[&str] = &[SCRIBE_V1, SCRIBE_V1_EXPERIMENTAL];
}

pub const DEFAULT_SPEECH: &str = speech::ELEVEN_MULTILINGUAL_V2;
pub const DEFAULT_TRANSCRIPTION: &str = transcription::SCRIBE_V1;
pub const DEFAULT_VOICE: &str = "21m00Tcm4TlvDq8ikWAM";
pub const ALL_SPEECH: &[&str] = speech::ALL;
pub const ALL_TRANSCRIPTION: &[&str] = transcription::ALL;

pub use speech::ELEVEN_MULTILINGUAL_V2;
pub use transcription::SCRIBE_V1;
