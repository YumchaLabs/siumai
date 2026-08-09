//! Curated ElevenLabs speech provider facade.

pub use siumai_provider_elevenlabs::configured::{
    ElevenLabsApiKey, ElevenLabsConfigError, ElevenLabsCredential, ElevenLabsCredentialError,
    ElevenLabsProfile, ElevenLabsProvider, ElevenLabsProviderBuilder, ElevenLabsSpeechModel,
    ElevenLabsTimestampGranularity, ElevenLabsTranscriptionFileFormat,
    ElevenLabsTranscriptionModel, ElevenLabsTranscriptionOptions,
};

pub mod models {
    pub use siumai_provider_elevenlabs::configured::models::{
        DEFAULT, DEFAULT_VOICE, ELEVEN_FLASH_V2, ELEVEN_FLASH_V2_5, ELEVEN_MULTILINGUAL_V1,
        ELEVEN_MULTILINGUAL_V2, ELEVEN_TURBO_V2, ELEVEN_TURBO_V2_5, ELEVEN_V3, SCRIBE_V1,
        SCRIBE_V2, SCRIBE_V2_REALTIME, VERIFIED, VERIFIED_TRANSCRIPTION,
    };
}

pub mod options {
    pub use siumai_provider_elevenlabs::configured::{
        ApplyTextNormalization, ElevenLabsPronunciationDictionaryLocator, ElevenLabsSpeechOptions,
        ElevenLabsTimestampGranularity, ElevenLabsTranscriptionFileFormat,
        ElevenLabsTranscriptionOptions, ElevenLabsVoiceSettings,
    };
}
