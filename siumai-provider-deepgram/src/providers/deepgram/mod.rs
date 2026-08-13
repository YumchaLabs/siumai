//! Transitional namespace for Deepgram-owned public types.

pub use crate::{
    DeepgramConfigError, DeepgramCredential, DeepgramCredentialError, DeepgramProvider,
    DeepgramProviderBuilder, DeepgramRedaction, DeepgramSummarizeOption,
    DeepgramTranscriptionModel, DeepgramTranscriptionOptions, VERSION,
};

pub mod models {
    pub use crate::models::*;
}
