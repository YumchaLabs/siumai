//! Dated Groq model advisories.
//!
//! These constants are completion hints verified against Groq's public model catalog on
//! 2026-08-05. They are not an allowlist; [`crate::GroqProvider`] accepts future model IDs.

/// Known language model IDs.
pub mod language {
    pub const LLAMA_3_1_8B_INSTANT: &str = "llama-3.1-8b-instant";
    pub const LLAMA_3_3_70B_VERSATILE: &str = "llama-3.3-70b-versatile";
    pub const GPT_OSS_20B: &str = "openai/gpt-oss-20b";
    pub const GPT_OSS_120B: &str = "openai/gpt-oss-120b";
    pub const GPT_OSS_SAFEGUARD_20B: &str = "openai/gpt-oss-safeguard-20b";
    pub const COMPOUND: &str = "groq/compound";
    pub const COMPOUND_MINI: &str = "groq/compound-mini";
    /// Retired on 2026-07-17; use [`GPT_OSS_120B`] instead.
    pub const QWEN3_32B: &str = "qwen/qwen3-32b";

    pub const ACTIVE: &[&str] = &[GPT_OSS_20B, GPT_OSS_120B, COMPOUND, COMPOUND_MINI];
    pub const DEPRECATED: &[&str] = &[LLAMA_3_1_8B_INSTANT, LLAMA_3_3_70B_VERSATILE];
    pub const RETIRED: &[&str] = &[QWEN3_32B];
    pub const KNOWN: &[&str] = &[
        LLAMA_3_1_8B_INSTANT,
        LLAMA_3_3_70B_VERSATILE,
        GPT_OSS_20B,
        GPT_OSS_120B,
        COMPOUND,
        COMPOUND_MINI,
        QWEN3_32B,
    ];

    /// Models verified for Groq's built-in browser-search tool on 2026-08-05.
    pub const BROWSER_SEARCH: &[&str] = &[GPT_OSS_20B, GPT_OSS_120B, GPT_OSS_SAFEGUARD_20B];

    /// Models verified for hosted browser search through Responses.
    pub const RESPONSES_BROWSER_SEARCH: &[&str] = BROWSER_SEARCH;
    /// Models verified for hosted code execution through Responses.
    pub const RESPONSES_CODE_EXECUTION: &[&str] = &[GPT_OSS_20B, GPT_OSS_120B];
}

/// Preview model IDs are completion hints and are intentionally absent from the stable catalog.
pub mod preview {
    pub const QWEN3_6_27B: &str = "qwen/qwen3.6-27b";
    pub const MINIMAX_M2_7: &str = "minimaxai/minimax-m2.7";
}

/// Known transcription model IDs.
pub mod transcription {
    pub const WHISPER_LARGE_V3: &str = "whisper-large-v3";
    pub const WHISPER_LARGE_V3_TURBO: &str = "whisper-large-v3-turbo";

    pub const KNOWN: &[&str] = &[WHISPER_LARGE_V3, WHISPER_LARGE_V3_TURBO];
}

/// Known Groq Orpheus speech model IDs.
pub mod speech {
    pub const ORPHEUS_V1_ENGLISH: &str = "canopylabs/orpheus-v1-english";
    pub const ORPHEUS_ARABIC_SAUDI: &str = "canopylabs/orpheus-arabic-saudi";

    pub const KNOWN: &[&str] = &[ORPHEUS_V1_ENGLISH, ORPHEUS_ARABIC_SAUDI];
}

pub const DEFAULT_SPEECH: &str = speech::ORPHEUS_V1_ENGLISH;
pub const CURRENT_SPEECH_MODELS: &[&str] = speech::KNOWN;

pub const VERIFIED_ON: &str = "2026-08-05";

pub(crate) fn is_known_language(model: &str) -> bool {
    language::KNOWN.contains(&model)
}

pub(crate) fn supports_browser_search(model: &str) -> bool {
    language::BROWSER_SEARCH.contains(&model)
}

pub(crate) fn supports_responses_browser_search(model: &str) -> bool {
    language::RESPONSES_BROWSER_SEARCH.contains(&model)
}

pub(crate) fn supports_responses_code_execution(model: &str) -> bool {
    language::RESPONSES_CODE_EXECUTION.contains(&model)
}

pub(crate) fn is_compound(model: &str) -> bool {
    matches!(model, language::COMPOUND | language::COMPOUND_MINI)
}

pub(crate) fn is_known_transcription(model: &str) -> bool {
    transcription::KNOWN.contains(&model)
}

pub(crate) fn is_known_speech(model: &str) -> bool {
    speech::KNOWN.contains(&model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalogs_are_hints_instead_of_defaults_or_allowlists() {
        assert!(is_known_language(language::GPT_OSS_20B));
        assert!(is_known_language(language::COMPOUND));
        assert!(supports_browser_search(language::GPT_OSS_120B));
        assert!(supports_browser_search(language::GPT_OSS_SAFEGUARD_20B));
        assert!(supports_responses_browser_search(
            language::GPT_OSS_SAFEGUARD_20B
        ));
        assert!(supports_responses_code_execution(language::GPT_OSS_20B));
        assert!(is_compound(language::COMPOUND_MINI));
        assert!(is_known_transcription(
            transcription::WHISPER_LARGE_V3_TURBO
        ));
        assert!(is_known_speech(speech::ORPHEUS_V1_ENGLISH));
        assert!(is_known_speech(speech::ORPHEUS_ARABIC_SAUDI));
        assert!(!is_known_language(preview::QWEN3_6_27B));
        assert!(!is_known_language("future-groq-model"));
    }
}
