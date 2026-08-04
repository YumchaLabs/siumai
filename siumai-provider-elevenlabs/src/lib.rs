//! siumai-provider-elevenlabs
//!
//! ElevenLabs provider implementation for speech synthesis and transcription.
#![deny(unsafe_code)]

#[allow(unused_imports)]
pub(crate) use siumai_provider_utils as provider_utils;

#[allow(unused_imports)]
pub(crate) use siumai_core::{
    LlmError, compat as core_compat, core, defaults, error, execution, retry, retry_api, speech,
    traits, transcription, types,
};

pub mod configured;
pub mod providers;

pub use providers::elevenlabs::*;
