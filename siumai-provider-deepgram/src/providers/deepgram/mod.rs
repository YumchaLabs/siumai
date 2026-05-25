//! `Deepgram` provider module.
//!
//! This module owns the AI SDK-aligned Deepgram speech/transcription surface.

pub mod client;
pub mod config;
pub mod ext;
pub mod models;
pub mod options;

pub use client::{DeepgramClient, DeepgramSpeechModel, DeepgramTranscriptionModel};
pub use config::DeepgramConfig;
pub use ext::{DeepgramSttRequestExt, DeepgramTtsRequestExt};
pub use options::{
    DeepgramSpeechModelOptions, DeepgramSpeechOptions, DeepgramSttOptions, DeepgramSummarizeOption,
    DeepgramTranscriptionModelOptions,
};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
