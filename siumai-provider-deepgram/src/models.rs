//! Deepgram model identifiers are conveniences, never an execution allowlist.

/// Deepgram prerecorded transcription identifiers verified against the native package contract.
pub mod transcription {
    pub const BASE: &str = "base";
    pub const BASE_GENERAL: &str = "base-general";
    pub const BASE_MEETING: &str = "base-meeting";
    pub const BASE_PHONECALL: &str = "base-phonecall";
    pub const BASE_FINANCE: &str = "base-finance";
    pub const BASE_CONVERSATIONALAI: &str = "base-conversationalai";
    pub const BASE_VOICEMAIL: &str = "base-voicemail";
    pub const BASE_VIDEO: &str = "base-video";
    pub const ENHANCED: &str = "enhanced";
    pub const ENHANCED_GENERAL: &str = "enhanced-general";
    pub const ENHANCED_MEETING: &str = "enhanced-meeting";
    pub const ENHANCED_PHONECALL: &str = "enhanced-phonecall";
    pub const ENHANCED_FINANCE: &str = "enhanced-finance";
    pub const NOVA: &str = "nova";
    pub const NOVA_GENERAL: &str = "nova-general";
    pub const NOVA_PHONECALL: &str = "nova-phonecall";
    pub const NOVA_MEDICAL: &str = "nova-medical";
    pub const NOVA_2: &str = "nova-2";
    pub const NOVA_2_GENERAL: &str = "nova-2-general";
    pub const NOVA_2_MEETING: &str = "nova-2-meeting";
    pub const NOVA_2_PHONECALL: &str = "nova-2-phonecall";
    pub const NOVA_2_FINANCE: &str = "nova-2-finance";
    pub const NOVA_2_CONVERSATIONALAI: &str = "nova-2-conversationalai";
    pub const NOVA_2_VOICEMAIL: &str = "nova-2-voicemail";
    pub const NOVA_2_VIDEO: &str = "nova-2-video";
    pub const NOVA_2_MEDICAL: &str = "nova-2-medical";
    pub const NOVA_2_DRIVETHRU: &str = "nova-2-drivethru";
    pub const NOVA_2_AUTOMOTIVE: &str = "nova-2-automotive";
    pub const NOVA_2_ATC: &str = "nova-2-atc";
    pub const NOVA_3: &str = "nova-3";
    pub const NOVA_3_GENERAL: &str = "nova-3-general";
    pub const NOVA_3_MEDICAL: &str = "nova-3-medical";

    pub const ALL: &[&str] = &[
        BASE,
        BASE_GENERAL,
        BASE_MEETING,
        BASE_PHONECALL,
        BASE_FINANCE,
        BASE_CONVERSATIONALAI,
        BASE_VOICEMAIL,
        BASE_VIDEO,
        ENHANCED,
        ENHANCED_GENERAL,
        ENHANCED_MEETING,
        ENHANCED_PHONECALL,
        ENHANCED_FINANCE,
        NOVA,
        NOVA_GENERAL,
        NOVA_PHONECALL,
        NOVA_MEDICAL,
        NOVA_2,
        NOVA_2_GENERAL,
        NOVA_2_MEETING,
        NOVA_2_PHONECALL,
        NOVA_2_FINANCE,
        NOVA_2_CONVERSATIONALAI,
        NOVA_2_VOICEMAIL,
        NOVA_2_VIDEO,
        NOVA_2_MEDICAL,
        NOVA_2_DRIVETHRU,
        NOVA_2_AUTOMOTIVE,
        NOVA_2_ATC,
        NOVA_3,
        NOVA_3_GENERAL,
        NOVA_3_MEDICAL,
    ];
}

pub const DEFAULT_TRANSCRIPTION: &str = transcription::NOVA_3;
pub const ALL_TRANSCRIPTION: &[&str] = transcription::ALL;

pub use transcription::NOVA_3;
