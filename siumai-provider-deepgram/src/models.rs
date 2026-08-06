//! Dated Deepgram prerecorded model hints. Model identifiers remain open input.

pub mod transcription {
    pub const NOVA_3: &str = "nova-3";
    pub const NOVA_3_GENERAL: &str = "nova-3-general";
    pub const NOVA_3_MEDICAL: &str = "nova-3-medical";
    pub const NOVA_2: &str = "nova-2";
    pub const NOVA_2_GENERAL: &str = "nova-2-general";
    pub const NOVA_2_MEETING: &str = "nova-2-meeting";
    pub const NOVA_2_PHONECALL: &str = "nova-2-phonecall";
    pub const NOVA_2_FINANCE: &str = "nova-2-finance";
    pub const NOVA_2_CONVERSATIONAL_AI: &str = "nova-2-conversationalai";
    pub const NOVA_2_VOICEMAIL: &str = "nova-2-voicemail";
    pub const NOVA_2_VIDEO: &str = "nova-2-video";
    pub const NOVA_2_MEDICAL: &str = "nova-2-medical";
    pub const NOVA_2_DRIVE_THRU: &str = "nova-2-drivethru";
    pub const NOVA_2_AUTOMOTIVE: &str = "nova-2-automotive";
    pub const NOVA_2_ATC: &str = "nova-2-atc";

    pub const CURRENT: &[&str] = &[
        NOVA_3,
        NOVA_3_GENERAL,
        NOVA_3_MEDICAL,
        NOVA_2,
        NOVA_2_GENERAL,
        NOVA_2_MEETING,
        NOVA_2_PHONECALL,
        NOVA_2_FINANCE,
        NOVA_2_CONVERSATIONAL_AI,
        NOVA_2_VOICEMAIL,
        NOVA_2_VIDEO,
        NOVA_2_MEDICAL,
        NOVA_2_DRIVE_THRU,
        NOVA_2_AUTOMOTIVE,
        NOVA_2_ATC,
    ];
}

pub const DEFAULT_TRANSCRIPTION: &str = transcription::NOVA_3;
pub const CURRENT_TRANSCRIPTION_MODELS: &[&str] = transcription::CURRENT;

pub use transcription::NOVA_3;

pub(crate) fn is_current(model: &str) -> bool {
    CURRENT_TRANSCRIPTION_MODELS.contains(&model)
}
