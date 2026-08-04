//! Curated Deepgram prerecorded transcription provider facade.

pub use siumai_provider_deepgram::{
    DeepgramConfigError, DeepgramCredential, DeepgramCredentialError, DeepgramProvider,
    DeepgramProviderBuilder, DeepgramTranscriptionModel,
};

pub mod models {
    pub use siumai_provider_deepgram::models::{ALL_TRANSCRIPTION, DEFAULT_TRANSCRIPTION, NOVA_3};
}

pub mod options {
    pub use siumai_provider_deepgram::{
        DeepgramDiarizeModel, DeepgramRedaction, DeepgramSummarizeOption,
        DeepgramTranscriptionOptions,
    };
}
