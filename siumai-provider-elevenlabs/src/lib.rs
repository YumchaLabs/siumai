//! siumai-provider-elevenlabs
//!
//! ElevenLabs provider implementation for speech synthesis.
#![deny(unsafe_code)]

pub mod configured;

pub use configured::{
    ApplyTextNormalization, ElevenLabsApiKey, ElevenLabsConfigError, ElevenLabsCredential,
    ElevenLabsCredentialError, ElevenLabsProfile, ElevenLabsPronunciationDictionaryLocator,
    ElevenLabsProvider, ElevenLabsProviderBuilder, ElevenLabsSpeechModel, ElevenLabsSpeechOptions,
    ElevenLabsVoiceSettings,
};
