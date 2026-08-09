//! Curated Deepgram prerecorded transcription provider facade.

pub use siumai_provider_deepgram::{
    DeepgramConfigError, DeepgramCredential, DeepgramCredentialError, DeepgramProfile,
    DeepgramProfileError, DeepgramProvider, DeepgramProviderBuilder, DeepgramSpeechModel,
    DeepgramTranscriptionModel,
};

pub mod models {
    pub use siumai_provider_deepgram::models::{
        AURA_2_THALIA_EN, CURRENT_SPEECH_MODELS, CURRENT_TRANSCRIPTION_MODELS, DEFAULT_SPEECH,
        DEFAULT_TRANSCRIPTION, NOVA_3, speech, transcription,
    };
}

pub mod options {
    pub use siumai_provider_deepgram::{
        DeepgramDiarizeModel, DeepgramRedaction, DeepgramSummarizeOption,
        DeepgramTranscriptionOptions,
    };
}
