//! Configured Deepgram provider for final-result transcription.
#![deny(unsafe_code)]

mod credential;
mod model;
pub mod models;
mod options;
mod profile;
mod provider;

pub mod providers;

pub use credential::{DeepgramCredential, DeepgramCredentialError};
pub use model::DeepgramTranscriptionModel;
pub use options::{
    DeepgramDiarizeModel, DeepgramRedaction, DeepgramSummarizeOption, DeepgramTranscriptionOptions,
};
pub use profile::{DeepgramProfile, DeepgramProfileError};
pub use provider::{DeepgramConfigError, DeepgramProvider, DeepgramProviderBuilder};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
