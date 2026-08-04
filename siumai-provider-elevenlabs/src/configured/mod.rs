//! Configured ElevenLabs native Text-to-Speech provider.

mod credentials;
mod model;
mod options;
mod policy;
mod profile;
mod provider;

pub mod models;

pub use credentials::{ElevenLabsApiKey, ElevenLabsCredential, ElevenLabsCredentialError};
pub use model::ElevenLabsSpeechModel;
pub use options::{
    ApplyTextNormalization, ElevenLabsPronunciationDictionaryLocator, ElevenLabsSpeechOptions,
    ElevenLabsVoiceSettings,
};
pub use profile::ElevenLabsProfile;
pub use provider::{ElevenLabsConfigError, ElevenLabsProvider, ElevenLabsProviderBuilder};
