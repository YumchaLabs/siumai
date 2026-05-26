//! `ElevenLabs` provider module.
//!
//! This module owns the AI SDK-aligned ElevenLabs speech/transcription surface.

pub mod client;
pub mod config;
pub mod ext;
pub mod models;
pub mod options;
pub mod voices;

pub use client::{ElevenLabsClient, ElevenLabsSpeechModel, ElevenLabsTranscriptionModel};
pub use config::ElevenLabsConfig;
pub use ext::{ElevenLabsSttRequestExt, ElevenLabsTtsRequestExt};
pub use options::{
    ApplyTextNormalization, ElevenLabsPronunciationDictionaryLocator, ElevenLabsSpeechModelOptions,
    ElevenLabsSpeechOptions, ElevenLabsSttOptions, ElevenLabsTranscriptionFileFormat,
    ElevenLabsTranscriptionModelOptions, ElevenLabsTranscriptionTimestampsGranularity,
    ElevenLabsVoiceSettings,
};
pub use voices::{
    ElevenLabsVerifiedLanguage, ElevenLabsVoice, ElevenLabsVoiceListQuery,
    ElevenLabsVoiceListResponse, ElevenLabsVoiceSettingsResponse, ElevenLabsVoices,
};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
