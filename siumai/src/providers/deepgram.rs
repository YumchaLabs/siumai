//! Curated Deepgram prerecorded transcription provider facade.

pub use siumai_provider_deepgram::{
    DeepgramConfigError, DeepgramCredential, DeepgramCredentialError, DeepgramProfile,
    DeepgramProfileError, DeepgramProvider, DeepgramProviderBuilder, DeepgramTranscriptionModel,
};

pub mod models {
    pub use siumai_provider_deepgram::models::{
        CURRENT_TRANSCRIPTION_MODELS, DEFAULT_TRANSCRIPTION, NOVA_3, transcription,
    };
}

pub mod options {
    pub use siumai_provider_deepgram::{
        DeepgramDiarizeModel, DeepgramRedaction, DeepgramSummarizeOption,
        DeepgramTranscriptionOptions,
    };
}
