//! Dated MiniMax model advisories.
//!
//! Model identifiers remain open input. These constants improve discovery and
//! preserve dated support evidence; they are not an execution allowlist.

/// Official source used for the current language-model advisory catalog.
pub const MODEL_CATALOG_SOURCE: &str =
    "https://platform.minimax.io/docs/api-reference/text-chat-anthropic";
/// Date on which the advisory catalog was verified.
pub const MODEL_CATALOG_VERIFIED_ON: &str = "2026-08-06";

pub const MINIMAX_M3: &str = "MiniMax-M3";
pub const MINIMAX_M2_7: &str = "MiniMax-M2.7";
pub const MINIMAX_M2_7_HIGHSPEED: &str = "MiniMax-M2.7-highspeed";
pub const MINIMAX_M2_5: &str = "MiniMax-M2.5";
pub const MINIMAX_M2_5_HIGHSPEED: &str = "MiniMax-M2.5-highspeed";
pub const MINIMAX_M2_1: &str = "MiniMax-M2.1";
pub const MINIMAX_M2_1_HIGHSPEED: &str = "MiniMax-M2.1-highspeed";
pub const MINIMAX_M2: &str = "MiniMax-M2";

pub const ALL_LANGUAGE: &[&str] = &[
    MINIMAX_M3,
    MINIMAX_M2_7,
    MINIMAX_M2_7_HIGHSPEED,
    MINIMAX_M2_5,
    MINIMAX_M2_5_HIGHSPEED,
    MINIMAX_M2_1,
    MINIMAX_M2_1_HIGHSPEED,
    MINIMAX_M2,
];

pub mod speech {
    pub const SPEECH_2_8_HD: &str = "speech-2.8-hd";
    pub const SPEECH_2_8_TURBO: &str = "speech-2.8-turbo";
    pub const SPEECH_2_6_HD: &str = "speech-2.6-hd";
    pub const SPEECH_2_6_TURBO: &str = "speech-2.6-turbo";
    pub const SPEECH_02_HD: &str = "speech-02-hd";
    pub const SPEECH_02_TURBO: &str = "speech-02-turbo";
    pub const SPEECH_01_HD: &str = "speech-01-hd";
    pub const SPEECH_01_TURBO: &str = "speech-01-turbo";
}

pub mod video {
    pub const MINIMAX_H3: &str = "MiniMax-H3";
}

pub mod music {
    pub const MUSIC_3_0: &str = "music-3.0";
    pub const MUSIC_3_0_FREE: &str = "music-3.0-free";
    pub const MUSIC_2_6: &str = "music-2.6";
    pub const MUSIC_2_6_FREE: &str = "music-2.6-free";
    pub const MUSIC_COVER: &str = "music-cover";
    pub const MUSIC_COVER_FREE: &str = "music-cover-free";
}

pub mod image {
    pub const IMAGE_01: &str = "image-01";
    pub const IMAGE_01_LIVE: &str = "image-01-live";
}
