//! Curated ElevenLabs speech provider facade.

pub use siumai_provider_elevenlabs::configured::{
    ElevenLabsApiKey, ElevenLabsConfigError, ElevenLabsCredential, ElevenLabsCredentialError,
    ElevenLabsProfile, ElevenLabsProvider, ElevenLabsProviderBuilder, ElevenLabsSpeechModel,
};

pub mod models {
    pub use siumai_provider_elevenlabs::configured::models::{
        DEFAULT, DEFAULT_VOICE, ELEVEN_FLASH_V2, ELEVEN_FLASH_V2_5, ELEVEN_MULTILINGUAL_V1,
        ELEVEN_MULTILINGUAL_V2, ELEVEN_TURBO_V2, ELEVEN_TURBO_V2_5, ELEVEN_V3, VERIFIED,
    };
}

pub mod options {
    pub use siumai_provider_elevenlabs::configured::{
        ApplyTextNormalization, ElevenLabsPronunciationDictionaryLocator, ElevenLabsSpeechOptions,
        ElevenLabsVoiceSettings,
    };
}
