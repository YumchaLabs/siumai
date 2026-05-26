pub use siumai_provider_elevenlabs::providers::elevenlabs::{
    ElevenLabsClient, ElevenLabsConfig, ElevenLabsSpeechModel, ElevenLabsTranscriptionModel,
    VERSION,
};
use siumai_registry::provider::SiumaiBuilder;

/// Curated ElevenLabs model constants aligned with the AI SDK speech/transcription package surface.
pub mod models {
    pub use siumai_provider_elevenlabs::providers::elevenlabs::models::{
        self as model_sets, ALL_SPEECH, ALL_TRANSCRIPTION, DEFAULT_SPEECH, DEFAULT_TRANSCRIPTION,
        DEFAULT_VOICE, speech, transcription,
    };
}

/// Create the unified ElevenLabs provider builder.
///
/// This mirrors the AI SDK package-level `elevenlabs` export while routing through Siumai's
/// registry-owned family handles.
pub fn elevenlabs() -> SiumaiBuilder {
    SiumaiBuilder::new().elevenlabs()
}

/// Create the unified ElevenLabs provider builder.
///
/// This is the Rust package-surface analogue of AI SDK `createElevenLabs()`.
pub fn create_elevenlabs() -> SiumaiBuilder {
    elevenlabs()
}

/// Typed provider options (`provider_options_map["elevenlabs"]`).
pub mod options {
    pub use siumai_provider_elevenlabs::providers::elevenlabs::{
        ApplyTextNormalization, ElevenLabsPronunciationDictionaryLocator,
        ElevenLabsSpeechModelOptions, ElevenLabsSpeechOptions, ElevenLabsSttOptions,
        ElevenLabsTranscriptionFileFormat, ElevenLabsTranscriptionModelOptions,
        ElevenLabsTranscriptionTimestampsGranularity, ElevenLabsVoiceSettings,
    };
}

/// Request extension traits for attaching ElevenLabs provider options.
pub mod ext {
    pub use siumai_provider_elevenlabs::providers::elevenlabs::{
        ElevenLabsSttRequestExt, ElevenLabsTtsRequestExt,
    };
}

/// Provider-specific resources not covered by the unified speech/transcription families.
pub mod resources {
    pub use siumai_provider_elevenlabs::providers::elevenlabs::{
        ElevenLabsCreateIvcVoiceRequest, ElevenLabsCreateIvcVoiceResponse,
        ElevenLabsCreatePronunciationDictionaryFromFileRequest,
        ElevenLabsCreatePronunciationDictionaryFromRulesRequest,
        ElevenLabsPronunciationDictionaries, ElevenLabsPronunciationDictionary,
        ElevenLabsPronunciationDictionaryCreateResponse,
        ElevenLabsPronunciationDictionaryDownloadResponse,
        ElevenLabsPronunciationDictionaryListQuery, ElevenLabsPronunciationDictionaryListResponse,
        ElevenLabsPronunciationDictionaryRule, ElevenLabsPronunciationDictionaryRuleRequest,
        ElevenLabsPronunciationDictionaryRulesMutationRequest,
        ElevenLabsPronunciationDictionaryRulesMutationResponse,
        ElevenLabsRemovePronunciationDictionaryRulesRequest,
        ElevenLabsUpdatePronunciationDictionaryRequest, ElevenLabsUpdateVoiceSettingsRequest,
        ElevenLabsVerifiedLanguage, ElevenLabsVoice, ElevenLabsVoiceListQuery,
        ElevenLabsVoiceListResponse, ElevenLabsVoiceSampleFile, ElevenLabsVoiceSettingsResponse,
        ElevenLabsVoiceSettingsUpdateResponse, ElevenLabsVoiceStatusResponse, ElevenLabsVoices,
    };
}

pub use ext::{ElevenLabsSttRequestExt, ElevenLabsTtsRequestExt};
pub use models::{
    ALL_SPEECH, ALL_TRANSCRIPTION, DEFAULT_SPEECH, DEFAULT_TRANSCRIPTION, DEFAULT_VOICE,
    model_sets, speech, transcription,
};
pub use options::{
    ApplyTextNormalization, ElevenLabsPronunciationDictionaryLocator, ElevenLabsSpeechModelOptions,
    ElevenLabsSpeechOptions, ElevenLabsSttOptions, ElevenLabsTranscriptionFileFormat,
    ElevenLabsTranscriptionModelOptions, ElevenLabsTranscriptionTimestampsGranularity,
    ElevenLabsVoiceSettings,
};
